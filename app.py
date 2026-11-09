"""
Local LLM server.

Features
  - Persistent chat sessions (SQLite)
  - File transfer: phone -> laptop (and back)
  - Project context: register a code folder, attach files to a session

Run:
    pip install fastapi uvicorn httpx python-multipart
    python -m uvicorn app:app --host 0.0.0.0 --port 8000

Then open http://<lan-ip>:8000
"""

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

BASE = Path(__file__).resolve().parent
DB_PATH = BASE / "server.db"
UPLOADS = BASE / "uploads"
UPLOADS.mkdir(exist_ok=True)

OLLAMA = "http://localhost:11434"
MODEL = "qwen2.5-coder:7b"

# Files worth pulling into context. Everything else is ignored.
CODE_EXT = {
    ".java", ".kt", ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs",
    ".sql", ".xml", ".yaml", ".yml", ".json", ".properties", ".md",
    ".sh", ".toml", ".gradle", ".html", ".css",
}
SKIP_DIRS = {
    ".git", "target", "build", "node_modules", ".idea", ".venv",
    "venv", "__pycache__", "dist", ".gradle", "out",
}
MAX_CONTEXT_BYTES = 120_000  # keep the prompt inside the model's window


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------

@contextmanager
def db():
    """Commit on clean exit, always close.

    sqlite3's own connection context manager commits but does NOT close, so
    using it directly leaks a file handle per request.
    """
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 10000")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with db() as c:
        # WAL lets a reader run while a write is in flight, which matters once
        # a phone and the laptop are both hitting this.
        c.execute("PRAGMA journal_mode = WAL")
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                name      TEXT NOT NULL,
                path      TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                title      TEXT NOT NULL,
                project_id INTEGER,
                context    TEXT NOT NULL DEFAULT '[]',
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id);
            """
        )


init_db()
app = FastAPI(title="Local LLM server")


# --------------------------------------------------------------------------
# sessions
# --------------------------------------------------------------------------

class NewSession(BaseModel):
    title: str = "New chat"
    project_id: int | None = None


@app.get("/api/sessions")
def list_sessions():
    with db() as c:
        rows = c.execute(
            "SELECT s.*, p.name AS project_name "
            "FROM sessions s LEFT JOIN projects p ON p.id = s.project_id "
            "ORDER BY s.created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


@app.post("/api/sessions")
def create_session(body: NewSession):
    with db() as c:
        cur = c.execute(
            "INSERT INTO sessions (title, project_id, created_at) VALUES (?,?,?)",
            (body.title, body.project_id, time.time()),
        )
        return {"id": cur.lastrowid}


@app.delete("/api/sessions/{sid}")
def delete_session(sid: int):
    with db() as c:
        c.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
        c.execute("DELETE FROM sessions WHERE id = ?", (sid,))
    return {"ok": True}


@app.get("/api/sessions/{sid}")
def get_session(sid: int):
    with db() as c:
        s = c.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()
        if not s:
            raise HTTPException(404, "no such session")
        msgs = c.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
            (sid,),
        ).fetchall()
    out = dict(s)
    out["context"] = json.loads(out["context"])
    out["messages"] = [dict(m) for m in msgs]
    return out


class ContextUpdate(BaseModel):
    files: list[str]


@app.put("/api/sessions/{sid}/context")
def set_context(sid: int, body: ContextUpdate):
    with db() as c:
        c.execute(
            "UPDATE sessions SET context = ? WHERE id = ?",
            (json.dumps(body.files), sid),
        )
    return {"ok": True, "files": body.files}


# --------------------------------------------------------------------------
# projects
# --------------------------------------------------------------------------

class NewProject(BaseModel):
    name: str
    path: str


@app.get("/api/projects")
def list_projects():
    with db() as c:
        return [dict(r) for r in c.execute("SELECT * FROM projects ORDER BY name")]


@app.post("/api/projects")
def create_project(body: NewProject):
    p = Path(body.path).expanduser()
    if not p.is_dir():
        raise HTTPException(400, f"not a directory: {p}")
    with db() as c:
        cur = c.execute(
            "INSERT INTO projects (name, path) VALUES (?,?)", (body.name, str(p))
        )
        return {"id": cur.lastrowid}


@app.delete("/api/projects/{pid}")
def delete_project(pid: int):
    with db() as c:
        c.execute("DELETE FROM projects WHERE id = ?", (pid,))
    return {"ok": True}


def project_root(pid: int) -> Path:
    with db() as c:
        row = c.execute("SELECT path FROM projects WHERE id = ?", (pid,)).fetchone()
    if not row:
        raise HTTPException(404, "no such project")
    return Path(row["path"])


@app.get("/api/projects/{pid}/files")
def project_files(pid: int, q: str = ""):
    """Relative paths of code files in the project, optionally filtered."""
    root = project_root(pid)
    out = []
    for f in root.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in CODE_EXT:
            continue
        if any(part in SKIP_DIRS for part in f.parts):
            continue
        rel = str(f.relative_to(root)).replace("\\", "/")
        if q and q.lower() not in rel.lower():
            continue
        out.append(rel)
        if len(out) >= 2000:
            break
    return sorted(out)


@app.get("/api/projects/{pid}/grep")
def project_grep(pid: int, q: str):
    """Files whose contents mention q. Useful for finding context to attach."""
    if not q.strip():
        return []
    root = project_root(pid)
    hits = []
    needle = q.lower()
    for f in root.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in CODE_EXT:
            continue
        if any(part in SKIP_DIRS for part in f.parts):
            continue
        try:
            if needle in f.read_text(encoding="utf-8", errors="ignore").lower():
                hits.append(str(f.relative_to(root)).replace("\\", "/"))
        except OSError:
            continue
        if len(hits) >= 50:
            break
    return sorted(hits)


def build_context_block(project_id: int | None, files: list[str]) -> str:
    if not project_id or not files:
        return ""
    root = project_root(project_id)
    parts, total = [], 0
    for rel in files:
        f = (root / rel).resolve()
        # refuse anything that escapes the project root
        if root.resolve() not in f.parents and f != root.resolve():
            continue
        if not f.is_file():
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if total + len(text) > MAX_CONTEXT_BYTES:
            text = text[: max(0, MAX_CONTEXT_BYTES - total)]
        total += len(text)
        parts.append(f"--- {rel} ---\n{text}")
        if total >= MAX_CONTEXT_BYTES:
            break
    if not parts:
        return ""
    return (
        "You are helping with the following codebase. Use these files as the "
        "source of truth; say so if something is not in them.\n\n"
        + "\n\n".join(parts)
    )


# --------------------------------------------------------------------------
# files (phone -> laptop)
# --------------------------------------------------------------------------

def safe_name(name: str) -> str:
    return Path(name).name.replace("..", "_")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), note: str = Form("")):
    dest = UPLOADS / safe_name(file.filename or "upload.bin")
    stem, suffix = dest.stem, dest.suffix
    i = 1
    while dest.exists():
        dest = UPLOADS / f"{stem}_{i}{suffix}"
        i += 1
    with dest.open("wb") as out:
        while chunk := await file.read(1 << 20):
            out.write(chunk)
    return {"name": dest.name, "size": dest.stat().st_size, "note": note}


@app.get("/api/files")
def list_files():
    out = []
    for f in sorted(UPLOADS.iterdir(), key=lambda p: -p.stat().st_mtime):
        if f.is_file():
            out.append({"name": f.name, "size": f.stat().st_size, "mtime": f.stat().st_mtime})
    return out


@app.get("/api/files/{name}")
def download(name: str):
    f = UPLOADS / safe_name(name)
    if not f.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(f, filename=f.name)


@app.delete("/api/files/{name}")
def delete_file(name: str):
    f = UPLOADS / safe_name(name)
    if f.is_file():
        f.unlink()
    return {"ok": True}


# --------------------------------------------------------------------------
# chat
# --------------------------------------------------------------------------

class ChatIn(BaseModel):
    session_id: int
    content: str


@app.post("/api/chat")
async def chat(body: ChatIn):
    with db() as c:
        s = c.execute("SELECT * FROM sessions WHERE id = ?", (body.session_id,)).fetchone()
        if not s:
            raise HTTPException(404, "no such session")
        history = [
            {"role": r["role"], "content": r["content"]}
            for r in c.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
                (body.session_id,),
            )
        ]
        c.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
            (body.session_id, "user", body.content, time.time()),
        )
        # name the session after its first message
        if not history:
            c.execute(
                "UPDATE sessions SET title = ? WHERE id = ?",
                (body.content[:60], body.session_id),
            )

    messages = []
    ctx = build_context_block(s["project_id"], json.loads(s["context"]))
    if ctx:
        messages.append({"role": "system", "content": ctx})
    messages.extend(history)
    messages.append({"role": "user", "content": body.content})

    async def stream():
        collected = []
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA}/api/chat",
                    json={"model": MODEL, "messages": messages, "stream": True},
                ) as r:
                    async for line in r.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        piece = data.get("message", {}).get("content", "")
                        if piece:
                            collected.append(piece)
                            yield piece
        finally:
            if collected:
                with db() as c:
                    c.execute(
                        "INSERT INTO messages (session_id, role, content, created_at) "
                        "VALUES (?,?,?,?)",
                        (body.session_id, "assistant", "".join(collected), time.time()),
                    )

    return StreamingResponse(stream(), media_type="text/plain")


@app.get("/api/health")
def health():
    return {"ok": True, "model": MODEL}


# --------------------------------------------------------------------------
# ui
# --------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


PAGE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Local LLM</title>
<style>
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;font-family:system-ui,sans-serif;background:#0d0d0f;color:#e8e8ea;
     height:100dvh;display:flex;overflow:hidden}
aside{width:260px;background:#141417;border-right:1px solid #26262b;
      display:flex;flex-direction:column;flex-shrink:0}
aside.hidden{display:none}
.side-h{padding:12px;border-bottom:1px solid #26262b;display:flex;gap:8px;align-items:center}
.side-h b{font-size:13px;flex:1}
.list{flex:1;overflow-y:auto;padding:8px}
.item{padding:9px 10px;border-radius:7px;font-size:13px;cursor:pointer;
      white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#b9b9c0}
.item:hover{background:#1d1d22}
.item.active{background:#2563eb;color:#fff}
main{flex:1;display:flex;flex-direction:column;min-width:0}
header{padding:10px 14px;border-bottom:1px solid #26262b;display:flex;
       gap:10px;align-items:center;font-size:13px}
header .title{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#log{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:12px}
.msg{max-width:min(760px,86%);padding:10px 14px;border-radius:12px;
     white-space:pre-wrap;line-height:1.5;font-size:14px}
.user{align-self:flex-end;background:#2563eb}
.bot{align-self:flex-start;background:#17171b;border:1px solid #26262b}
form.ask{display:flex;gap:8px;padding:12px;border-top:1px solid #26262b}
input,select,textarea{background:#17171b;color:#e8e8ea;border:1px solid #2e2e35;
      border-radius:8px;padding:10px;font-size:16px;font-family:inherit}
input:focus,select:focus{outline:1px solid #2563eb}
form.ask input{flex:1}
button{background:#2563eb;color:#fff;border:0;border-radius:8px;padding:10px 16px;
       font-size:14px;cursor:pointer}
button.ghost{background:#24242b;color:#c9c9d1}
button:disabled{opacity:.45;cursor:default}
dialog{background:#141417;color:#e8e8ea;border:1px solid #2e2e35;border-radius:12px;
       padding:0;width:min(560px,92vw)}
dialog::backdrop{background:#000a}
.dlg-h{padding:12px 14px;border-bottom:1px solid #26262b;font-weight:600;font-size:14px}
.dlg-b{padding:14px;display:flex;flex-direction:column;gap:10px;max-height:60vh;overflow:auto}
.dlg-f{padding:12px 14px;border-top:1px solid #26262b;display:flex;gap:8px;justify-content:flex-end}
label.row{display:flex;gap:8px;align-items:center;font-size:13px;color:#c2c2ca}
.fileline{display:flex;gap:8px;align-items:center;font-size:13px;
          padding:7px 9px;background:#17171b;border-radius:7px}
.fileline span{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.muted{color:#7e7e8a;font-size:12px}
.chips{display:flex;flex-wrap:wrap;gap:6px;padding:0 14px 8px}
.chip{background:#24242b;border:1px solid #32323a;border-radius:999px;
      padding:3px 9px;font-size:11px;color:#b9b9c0}
@media(max-width:720px){aside{position:absolute;z-index:5;height:100%}}
</style>
</head>
<body>

<aside id="side">
  <div class="side-h">
    <b>Sessions</b>
    <button class="ghost" onclick="newSession()">+</button>
  </div>
  <div class="list" id="sessions"></div>
  <div class="side-h" style="border-top:1px solid #26262b;border-bottom:0">
    <button class="ghost" style="flex:1" onclick="filesDlg.showModal();loadFiles()">Files</button>
    <button class="ghost" style="flex:1" onclick="openProjects()">Projects</button>
  </div>
</aside>

<main>
  <header>
    <button class="ghost" onclick="side.classList.toggle('hidden')">☰</button>
    <div class="title" id="title">No session</div>
    <button class="ghost" id="ctxBtn" onclick="openContext()">Context</button>
  </header>
  <div class="chips" id="chips"></div>
  <div id="log"></div>
  <form class="ask" id="ask">
    <input id="q" placeholder="Ask something..." autocomplete="off" required>
    <button id="send">Send</button>
  </form>
</main>

<!-- files -->
<dialog id="filesDlg">
  <div class="dlg-h">Files</div>
  <div class="dlg-b">
    <form id="upForm">
      <input type="file" id="upInput" multiple>
    </form>
    <div class="muted">Uploads land in ./uploads on the laptop.</div>
    <div id="fileList" style="display:flex;flex-direction:column;gap:6px"></div>
  </div>
  <div class="dlg-f"><button class="ghost" onclick="filesDlg.close()">Close</button></div>
</dialog>

<!-- projects -->
<dialog id="projDlg">
  <div class="dlg-h">Projects</div>
  <div class="dlg-b">
    <div id="projList" style="display:flex;flex-direction:column;gap:6px"></div>
    <div class="muted">Add a folder on the laptop, e.g. C:\work\project-x</div>
    <input id="projName" placeholder="Name">
    <input id="projPath" placeholder="Absolute path">
    <button onclick="addProject()">Add project</button>
  </div>
  <div class="dlg-f"><button class="ghost" onclick="projDlg.close()">Close</button></div>
</dialog>

<!-- context -->
<dialog id="ctxDlg">
  <div class="dlg-h">Session context</div>
  <div class="dlg-b">
    <select id="ctxProject" onchange="onProjectChange()"></select>
    <input id="ctxFilter" placeholder="Filter by path" oninput="loadProjectFiles()">
    <input id="ctxGrep" placeholder="Or search file contents, then press Enter">
    <div class="muted" id="ctxCount"></div>
    <div id="ctxFiles" style="display:flex;flex-direction:column;gap:4px"></div>
  </div>
  <div class="dlg-f">
    <button class="ghost" onclick="ctxDlg.close()">Cancel</button>
    <button onclick="saveContext()">Save</button>
  </div>
</dialog>

<script>
const $ = (id) => document.getElementById(id);
const log = $('log'), side = $('side');
let sid = null, session = null, projects = [], ctxSelected = new Set();

const api = async (url, opts) => {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
};

/* ---------- sessions ---------- */
async function loadSessions(){
  const rows = await api('/api/sessions');
  $('sessions').innerHTML = '';
  rows.forEach(s => {
    const d = document.createElement('div');
    d.className = 'item' + (s.id === sid ? ' active' : '');
    d.textContent = s.title;
    d.onclick = () => openSession(s.id);
    d.oncontextmenu = async (e) => {
      e.preventDefault();
      if (confirm('Delete this session?')) {
        await fetch('/api/sessions/' + s.id, {method:'DELETE'});
        if (sid === s.id) { sid = null; session = null; render(); }
        loadSessions();
      }
    };
    $('sessions').appendChild(d);
  });
}

async function newSession(){
  const r = await api('/api/sessions', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({title:'New chat'})
  });
  await openSession(r.id);
  await loadSessions();
}

async function openSession(id){
  sid = id;
  session = await api('/api/sessions/' + id);
  render();
  loadSessions();
  if (window.innerWidth <= 720) side.classList.add('hidden');
}

function render(){
  log.innerHTML = '';
  $('title').textContent = session ? session.title : 'No session';
  $('chips').innerHTML = '';
  if (session) {
    (session.context || []).forEach(f => {
      const c = document.createElement('div');
      c.className = 'chip'; c.textContent = f;
      $('chips').appendChild(c);
    });
    session.messages.forEach(m => bubble(m.role === 'user' ? 'user' : 'bot', m.content));
  }
}

function bubble(cls, text){
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  d.textContent = text;
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
  return d;
}

/* ---------- chat ---------- */
$('ask').onsubmit = async (e) => {
  e.preventDefault();
  if (!sid) await newSession();
  const text = $('q').value.trim();
  if (!text) return;
  $('q').value = '';
  $('send').disabled = true;

  bubble('user', text);
  const out = bubble('bot', '…');
  let full = '';

  try {
    const res = await fetch('/api/chat', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({session_id: sid, content: text})
    });
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      full += dec.decode(value, {stream:true});
      out.textContent = full;
      log.scrollTop = log.scrollHeight;
    }
  } catch (err) {
    out.textContent = 'Error: ' + err.message;
  }

  $('send').disabled = false;
  $('q').focus();
  loadSessions();
};

/* ---------- files ---------- */
async function loadFiles(){
  const rows = await api('/api/files');
  const box = $('fileList');
  box.innerHTML = '';
  if (!rows.length) box.innerHTML = '<div class="muted">No files yet.</div>';
  rows.forEach(f => {
    const d = document.createElement('div');
    d.className = 'fileline';
    const kb = (f.size/1024).toFixed(0);
    d.innerHTML = `<span>${f.name}</span><span class="muted">${kb} KB</span>`;
    const dl = document.createElement('button');
    dl.className = 'ghost'; dl.textContent = '↓';
    dl.onclick = () => location.href = '/api/files/' + encodeURIComponent(f.name);
    const rm = document.createElement('button');
    rm.className = 'ghost'; rm.textContent = '✕';
    rm.onclick = async () => {
      await fetch('/api/files/' + encodeURIComponent(f.name), {method:'DELETE'});
      loadFiles();
    };
    d.append(dl, rm);
    box.appendChild(d);
  });
}

$('upInput').onchange = async (e) => {
  for (const f of e.target.files) {
    const fd = new FormData();
    fd.append('file', f);
    await fetch('/api/upload', {method:'POST', body: fd});
  }
  e.target.value = '';
  loadFiles();
};

/* ---------- projects ---------- */
async function openProjects(){
  projects = await api('/api/projects');
  const box = $('projList');
  box.innerHTML = '';
  if (!projects.length) box.innerHTML = '<div class="muted">No projects yet.</div>';
  projects.forEach(p => {
    const d = document.createElement('div');
    d.className = 'fileline';
    d.innerHTML = `<span>${p.name}<br><span class="muted">${p.path}</span></span>`;
    const rm = document.createElement('button');
    rm.className = 'ghost'; rm.textContent = '✕';
    rm.onclick = async () => {
      await fetch('/api/projects/' + p.id, {method:'DELETE'});
      openProjects();
    };
    d.append(rm);
    box.appendChild(d);
  });
  projDlg.showModal();
}

async function addProject(){
  try {
    await api('/api/projects', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({name: $('projName').value, path: $('projPath').value})
    });
    $('projName').value = ''; $('projPath').value = '';
    openProjects();
  } catch (e) { alert(e.message); }
}

/* ---------- context ---------- */
async function openContext(){
  if (!sid) { alert('Open a session first'); return; }
  projects = await api('/api/projects');
  const sel = $('ctxProject');
  sel.innerHTML = '<option value="">No project</option>';
  projects.forEach(p => {
    const o = document.createElement('option');
    o.value = p.id; o.textContent = p.name;
    if (p.id === session.project_id) o.selected = true;
    sel.appendChild(o);
  });
  ctxSelected = new Set(session.context || []);
  await loadProjectFiles();
  ctxDlg.showModal();
}

function onProjectChange(){
  ctxSelected = new Set();   // paths belong to the old project, drop them
  loadProjectFiles();
}

async function loadProjectFiles(){
  const pid = $('ctxProject').value;
  const box = $('ctxFiles');
  box.innerHTML = '';
  if (!pid) { box.innerHTML = '<div class="muted">Pick a project.</div>'; return; }
  const q = encodeURIComponent($('ctxFilter').value);
  const files = await api(`/api/projects/${pid}/files?q=${q}`);
  paintFiles(files);
}

$('ctxGrep').onkeydown = async (e) => {
  if (e.key !== 'Enter') return;
  e.preventDefault();
  const pid = $('ctxProject').value;
  if (!pid) return;
  const files = await api(`/api/projects/${pid}/grep?q=${encodeURIComponent(e.target.value)}`);
  paintFiles(files);
};

function paintFiles(files){
  const box = $('ctxFiles');
  box.innerHTML = '';
  updateCtxCount();
  if (!files.length) { box.innerHTML = '<div class="muted">Nothing matched.</div>'; return; }
  files.slice(0, 400).forEach(f => {
    const l = document.createElement('label');
    l.className = 'row';
    const cb = document.createElement('input');
    cb.type = 'checkbox'; cb.value = f; cb.checked = ctxSelected.has(f);
    cb.style.flex = '0 0 auto';
    // Track in a Set, not the DOM: filtering re-renders the list, and a file
    // selected under an earlier filter must survive that.
    cb.onchange = () => {
      cb.checked ? ctxSelected.add(f) : ctxSelected.delete(f);
      updateCtxCount();
    };
    l.append(cb, document.createTextNode(f));
    box.appendChild(l);
  });
}

function updateCtxCount(){
  const n = ctxSelected.size;
  $('ctxCount').textContent = n ? `${n} file${n > 1 ? 's' : ''} selected` : 'No files selected';
}

async function saveContext(){
  const pid = $('ctxProject').value;
  const files = [...ctxSelected];
  await api(`/api/sessions/${sid}/context`, {
    method:'PUT', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({files})
  });
  // persist project choice on the session
  await fetch(`/api/sessions/${sid}/project?pid=${pid || ''}`, {method:'PUT'});
  session = await api('/api/sessions/' + sid);
  render();
  ctxDlg.close();
}

loadSessions();
</script>
</body>
</html>
"""


# small endpoint the UI uses to bind a project to a session
@app.put("/api/sessions/{sid}/project")
def set_project(sid: int, pid: str = ""):
    val = int(pid) if pid.strip() else None
    with db() as c:
        c.execute("UPDATE sessions SET project_id = ? WHERE id = ?", (val, sid))
    return {"ok": True}