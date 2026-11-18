const $ = (id) => document.getElementById(id);
const log = $('log'), q = $('q');
let modelName = '', sid = null, session = null, curProject = null, projects = [], busy = false;

/* ---------- helpers ---------- */
// Build DOM without innerHTML so names/paths/model output can never inject markup.
function el(tag, props = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === 'class') n.className = v;
    else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  n.append(...kids);
  return n;
}

function toast(msg, kind = 'error') {
  const t = el('div', {class: 'toast ' + kind, role: kind === 'error' ? 'alert' : 'status'}, msg);
  $('toasts').appendChild(t);
  setTimeout(() => t.remove(), kind === 'error' ? 8000 : 4000);
}

// Fallbacks for a failed response that carries no message of its own.
const HINTS = {
  413: 'Too large.',
  502: 'Ollama did not answer properly.',
  400: 'Invalid request.',
};

// Turn a failed response into a readable message (FastAPI sends {"detail": ...},
// and the server words it for people, so it is shown as is).
async function errorText(r) {
  let detail = '';
  try {
    const j = await r.json();
    detail = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail || j);
  } catch (e) { /* not JSON */ }
  return detail || HINTS[r.status] || `HTTP ${r.status}`;
}

// An Error carrying the HTTP status, so callers can tell a refused message from an outage.
async function httpError(r) {
  const e = new Error(await errorText(r));
  e.status = r.status;
  return e;
}

async function request(url, opts) {
  let r;
  try { r = await fetch(url, opts); }
  catch (e) { throw new Error('Cannot reach the server. Check the network connection.'); }
  if (!r.ok) throw await httpError(r);
  return r;
}
const api = async (url, opts) => (await request(url, opts)).json();
const json = (method, body) => ({method, headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
const guard = (fn) => async (...a) => { try { return await fn(...a); } catch (e) { toast(e.message); } };

/* ---------- theme ---------- */
const media = window.matchMedia('(prefers-color-scheme: dark)');
const isDark = () => {
  const t = document.documentElement.dataset.theme;
  return t ? t === 'dark' : media.matches;
};
function paintThemeBtn() { $('themeBtn').textContent = isDark() ? '☀' : '☾'; }
$('themeBtn').onclick = () => {
  const next = isDark() ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem('theme', next); } catch (e) { /* private mode */ }
  paintThemeBtn();
};
media.addEventListener('change', paintThemeBtn);
paintThemeBtn();

/* ---------- safe markdown ---------- */
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;').replace(/'/g, '&#39;');

// Input is escaped first, so only the tags produced here can reach innerHTML.
function inline(s) {
  const codes = [];
  s = esc(s).replace(/`([^`\n]+)`/g, (_, c) => { codes.push(c); return '\u0000' + (codes.length - 1) + '\u0000'; });
  s = s.replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
    .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?!\w)/g, '$1<em>$2</em>');
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => '<code>' + codes[i] + '</code>');
}

function markdown(src) {
  const lines = src.replace(/\r\n?/g, '\n').split('\n');
  const out = [];
  let i = 0;
  const isBlockStart = (l) => /^\s*(```|#{1,4}\s|[-*+]\s|\d+[.)]\s|>)/.test(l);
  while (i < lines.length) {
    const line = lines[i];
    let m;
    if ((m = line.match(/^\s*```\s*([\w+#.-]*)/))) {
      const code = [];
      i++;
      while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) code.push(lines[i++]);
      i++; // closing fence (an unclosed fence, e.g. mid-stream, runs to the end)
      out.push(`<div class="code"><div class="code-h"><span>${esc(m[1] || 'code')}</span>` +
        `<button type="button" class="ghost copy">Copy</button></div><pre><code>${esc(code.join('\n'))}</code></pre></div>`);
    } else if ((m = line.match(/^(#{1,4})\s+(.*)/))) {
      out.push(`<h${m[1].length}>${inline(m[2])}</h${m[1].length}>`);
      i++;
    } else if (/^\s*([-*+]|\d+[.)])\s/.test(line)) {
      const ordered = /^\s*\d/.test(line), items = [];
      while (i < lines.length && /^\s*([-*+]|\d+[.)])\s/.test(lines[i]))
        items.push(`<li>${inline(lines[i++].replace(/^\s*([-*+]|\d+[.)])\s+/, ''))}</li>`);
      out.push(ordered ? `<ol>${items.join('')}</ol>` : `<ul>${items.join('')}</ul>`);
    } else if (/^\s*>/.test(line)) {
      const quote = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) quote.push(lines[i++].replace(/^\s*>\s?/, ''));
      out.push(`<blockquote>${inline(quote.join(' '))}</blockquote>`);
    } else if (!line.trim()) {
      i++;
    } else {
      const para = [];
      while (i < lines.length && lines[i].trim() && (!para.length || !isBlockStart(lines[i]))) para.push(lines[i++]);
      out.push(`<p>${para.map(inline).join('<br>')}</p>`);
    }
  }
  return out.join('');
}

function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
  // plain-http LAN pages have no Clipboard API
  const ta = el('textarea', {'aria-hidden': 'true', style: 'position:fixed;opacity:0'});
  ta.value = text;
  document.body.appendChild(ta);
  ta.select();
  const ok = document.execCommand('copy');
  ta.remove();
  return ok ? Promise.resolve() : Promise.reject(new Error('copy failed'));
}

log.addEventListener('click', async (e) => {
  const btn = e.target.closest('button.copy');
  if (!btn) return;
  try {
    await copyText(btn.closest('.code').querySelector('code').textContent);
    btn.textContent = 'Copied';
  } catch (err) { btn.textContent = 'Failed'; }
  setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
});

/* ---------- navigation ---------- */
function toggleNav(open) {
  document.body.classList.toggle('nav-open', open);
  if (open) $('side').querySelector('button, a[href], input, select, textarea')?.focus();
  else if ($('side').contains(document.activeElement)) $('menuBtn').focus();
}
$('menuBtn').onclick = () => toggleNav(true);
$('scrim').onclick = () => toggleNav(false);
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') toggleNav(false); });

// Exactly one of the two views is visible: a project workspace or a conversation.
function showView(name) {
  $('projectView').hidden = name !== 'project';
  $('chatView').hidden = name !== 'chat';
}

const activeProjectId = () => {
  if (!$('projectView').hidden && curProject) return curProject.id;
  return session ? session.project_id : null;
};

const refreshLists = () => { loadSessions(); loadProjects(); refreshProject(); };

/* ---------- chats ---------- */
// Render a list of chat rows (normal chats in the sidebar, or a project's chats).
function sessionList(box, rows, emptyText) {
  if (box.querySelector('input.rename')) return;  // do not wipe a title being edited
  box.replaceChildren();
  if (!rows.length) box.append(el('div', {class: 'muted', style: 'padding:8px 10px'}, emptyText));
  rows.forEach((s) => {
    const del = async () => {
      if (!confirm('Delete this chat?')) return;
      try { await request('/api/sessions/' + s.id, {method: 'DELETE'}); }
      catch (e) { toast(e.message); return; }
      if (sid === s.id) {
        sid = null; session = null; render();
        if (s.project_id) { openProject(s.project_id); return; }
      }
      refreshLists();
    };
    const open = el('button', {class: 'open', type: 'button', onclick: () => openSession(s.id)}, s.title);
    if (s.id === sid) open.setAttribute('aria-current', 'true');
    const ren = el('button', {class: 'icon ren', type: 'button', 'aria-label': 'Rename ' + s.title, title: 'Rename',
      onclick: () => editTitle(open, s.title, (t) => renameSession(s, t), ren)}, '✎');
    box.appendChild(el('div', {class: 'item' + (s.id === sid ? ' active' : ''), oncontextmenu: (e) => { e.preventDefault(); del(); }},
      open, ren, el('button', {class: 'icon del danger', type: 'button', 'aria-label': 'Delete ' + s.title, title: 'Delete', onclick: del}, '✕')));
  });
}

// Swap a title for a text field: Enter saves, Escape cancels, leaving the field saves.
// ``save(newTitle)`` does the work; focus returns to ``focusBack`` after a key press.
function editTitle(target, current, save, focusBack) {
  if (target.parentNode.querySelector('input.rename')) return;
  const input = el('input', {class: 'rename', type: 'text', value: current, maxlength: '200', 'aria-label': 'Chat title'});
  let done = false;
  const finish = async (commit, byKey) => {
    if (done) return;
    done = true;
    const next = input.value.trim();
    input.remove();
    target.hidden = false;
    if (byKey && focusBack) focusBack.focus();
    if (commit && next && next !== current) await save(next);
    refreshLists();
  };
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.isComposing) { e.preventDefault(); finish(true, true); }
    else if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); finish(false, true); }
  });
  input.addEventListener('blur', () => finish(true, false));
  target.hidden = true;
  target.after(input);
  input.focus();
  input.select();
}

async function renameSession(s, title) {
  try {
    const r = await api('/api/sessions/' + s.id, json('PUT', {title}));
    if (session && session.id === s.id) { session.title = r.title; $('title').textContent = r.title; }
  } catch (e) { toast(e.message); }
}

$('renameBtn').onclick = () => {
  if (session) editTitle($('title'), session.title, (t) => renameSession(session, t), $('renameBtn'));
};
$('title').addEventListener('dblclick', () => $('renameBtn').click());

const loadSessions = guard(async () => {
  sessionList($('sessions'), await api('/api/sessions'), 'No chats yet.');
});

// A chat inside a project when projectId is given, else a normal chat.
const newSession = guard(async (projectId) => {
  const r = projectId
    ? await api(`/api/projects/${projectId}/sessions`, json('POST', {title: 'New chat'}))
    : await api('/api/sessions', json('POST', {title: 'New chat'}));
  await openSession(r.id);
});

const openSession = guard(async (id) => {
  sid = id;
  session = await api('/api/sessions/' + id);
  render();
  refreshLists();
  toggleNav(false);
});

function render() {
  showView('chat');
  log.replaceChildren($('empty'));
  $('title').textContent = session ? session.title : 'New chat';
  $('title').title = session ? 'Double-click to rename' : '';
  $('renameBtn').hidden = !session;
  $('sub').textContent = modelName;
  const inProject = !!(session && session.project_id);
  $('projLink').hidden = !inProject;
  if (inProject) $('projLink').textContent = '← ' + (session.project_name || 'Project');
  $('emptyText').textContent = inProject
    ? 'This chat shares the files uploaded to its project. Relevant parts are retrieved for every question.'
    : 'Chats stay on this machine. Create a project to ground answers in your own files, PDFs and images.';
  if (session) session.messages.forEach((m) => bubble(m.role === 'user' ? 'user' : 'bot', m.content));
  $('empty').hidden = !!(session && session.messages.length);
}

function setBody(node, cls, text) {
  if (cls === 'user') node.textContent = text;
  else node.innerHTML = markdown(text);  // escaped by markdown() first
}

function bubble(cls, text) {
  $('empty').hidden = true;
  const d = el('div', {class: 'msg ' + cls});
  setBody(d, cls, text);
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
  return d;
}

/* ---------- chat ---------- */
let maxChars = 20000;  // the server's message limit (replaced by /api/health)
let abortCtl = null;   // aborts the reply being streamed (the Stop button)

function resizeInput() {
  q.style.height = 'auto';
  q.style.height = Math.min(q.scrollHeight, 200) + 'px';
}

// Live character counter, shown once the message is within 10% of the limit.
function updateCount() {
  const n = q.value.length, over = n > maxChars;
  $('count').hidden = n < maxChars * 0.9;
  $('count').textContent = `${n.toLocaleString()} / ${maxChars.toLocaleString()}` + (over ? ' (too long)' : '');
  $('count').classList.toggle('over', over);
  q.setAttribute('aria-invalid', over ? 'true' : 'false');
}

q.addEventListener('input', () => { resizeInput(); updateCount(); });
q.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); $('ask').requestSubmit(); }
});

function setBusy(on) {
  busy = on;
  $('send').hidden = on;
  $('stop').hidden = !on;
}

$('ask').onsubmit = (e) => {
  e.preventDefault();
  const text = q.value.trim();
  if (!text || busy) return;
  if (text.length > maxChars) {
    toast(`Message too long (${text.length} characters, max ${maxChars}). Shorten it to send.`);
    return;
  }
  send(text, true);
};

// Send one message and stream the reply. ``fromInput``: the text came from the
// input box, which is cleared once a chat exists to send it in.
async function send(text, fromInput) {
  if (busy) return;
  setBusy(true);
  try {
    if (!sid) await newSession();
    if (!sid) return;
    if (fromInput) { q.value = ''; resizeInput(); updateCount(); }
    log.querySelectorAll('button.retry').forEach((b) => b.remove());  // only the latest failure can be retried
    await stream(text, bubble('user', text));
  } finally {
    setBusy(false);
    q.focus();
    refreshLists();
  }
}

$('stop').onclick = () => { if (abortCtl) abortCtl.abort(); };

// The server tells how it built the prompt in response headers.
function sourcesFrom(res) {
  try {
    const v = JSON.parse(res.headers.get('X-RAG-Sources') || '[]');
    return Array.isArray(v) ? v.filter((x) => typeof x === 'string') : [];
  } catch (e) { return []; }
}

function notesFrom(res) {
  const sources = sourcesFrom(res);
  return {
    sources,
    status: res.headers.get('X-RAG-Status') || (sources.length ? 'used' : ''),
    trimmed: parseInt(res.headers.get('X-History-Trimmed') || '0', 10) || 0,
  };
}

const RAG_NOTES = {
  none: 'No project files matched this question',
  empty: 'This project has no searchable files yet',
  failed: 'Could not search project files',
};

// Small print under an answer: which project files it drew on, trimmed history, stopped.
function showNotes(after, info, stopped) {
  const lines = [];
  if (info.status === 'used' && info.sources.length) lines.push('Used project files: ' + info.sources.join(', '));
  else if (RAG_NOTES[info.status]) lines.push(RAG_NOTES[info.status]);
  if (info.trimmed > 0) lines.push(`Earlier messages are not sent to the model (${info.trimmed})`);
  if (stopped) lines.push('Stopped');
  if (!lines.length) return;
  after.after(el('div', {class: 'sources', title: lines.join('. ')}, ...lines.map((t) => el('div', {}, t))));
}

// Statuses where the server refused the message before storing anything.
const REFUSED = [400, 404, 413];

// No reply came. A refused message goes back into the input box (after any draft already there); otherwise the
// failure is shown with Retry (the server dropped the unanswered question, so
// resending does not duplicate it in the history).
function failReply(out, mine, text, msg, refused) {
  toast(msg);
  if (refused) {
    out.remove();
    mine.remove();
    // the box may already hold the next message: keep both rather than overwrite either
    q.value = q.value.trim() ? q.value.replace(/\s+$/, '') + '\n\n' + text : text;
    resizeInput(); updateCount();
    return;
  }
  out.classList.add('err');
  out.replaceChildren(msg + ' ', el('button', {class: 'ghost retry', type: 'button', onclick: () => {
    out.remove();
    mine.remove();
    send(text, false);
  }}, 'Retry'));
}

async function stream(text, mine) {
  const out = bubble('bot', '');
  out.innerHTML = '<span class="typing" role="status" aria-label="Assistant is typing"><i></i><i></i><i></i></span>';
  let full = '', queued = false, info = {sources: [], status: '', trimmed: 0};
  const paint = () => {
    queued = false;
    setBody(out, 'bot', full);
    log.scrollTop = log.scrollHeight;
  };
  const ctl = new AbortController();
  abortCtl = ctl;
  try {
    const res = await fetch('/api/chat', {...json('POST', {session_id: sid, content: text}), signal: ctl.signal});
    if (!res.ok) throw await httpError(res);
    info = notesFrom(res);
    const reader = res.body.getReader(), dec = new TextDecoder();
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      full += dec.decode(value, {stream: true});
      if (!queued) { queued = true; requestAnimationFrame(paint); }
    }
    paint();
    if (!full.trim()) throw new Error('The model returned an empty reply.');
    showNotes(out, info, false);
  } catch (err) {
    const stopped = err.name === 'AbortError';
    if (full.trim()) {  // keep what arrived; the server stores the partial reply too
      paint();
      showNotes(out, info, stopped);
      if (!stopped) toast(err.message);
    } else {
      const msg = stopped ? 'Stopped before any reply.'
        : err instanceof TypeError ? 'Cannot reach the server. Check the network connection.' : err.message;
      failReply(out, mine, text, msg, REFUSED.includes(err.status));
    }
  } finally {
    abortCtl = null;
  }
}

/* ---------- dialogs ---------- */
document.querySelectorAll('dialog').forEach((d) => {
  d.querySelectorAll('[data-close]').forEach((b) => b.addEventListener('click', () => d.close()));
  d.addEventListener('click', (e) => { if (e.target === d) d.close(); });  // backdrop click
});
$('newBtn').onclick = () => newSession();
$('filesBtn').onclick = () => { $('filesDlg').showModal(); loadFiles(); };

/* ---------- file transfer (general, to ./uploads) ---------- */
function fmtSize(n) {
  return n >= 1048576 ? (n / 1048576).toFixed(1) + ' MB' : Math.max(1, Math.round(n / 1024)) + ' KB';
}

const loadFiles = guard(async () => {
  const rows = await api('/api/files');
  const box = $('fileList');
  box.replaceChildren();
  if (!rows.length) box.append(el('div', {class: 'muted'}, 'No files yet.'));
  rows.forEach((f) => {
    box.append(el('div', {class: 'line'},
      el('span', {class: 'grow', title: f.name}, f.name),
      el('span', {class: 'muted'}, fmtSize(f.size)),
      el('button', {class: 'ghost', 'aria-label': 'Download ' + f.name, onclick: () => { location.href = '/api/files/' + encodeURIComponent(f.name); }}, '↓'),
      el('button', {class: 'ghost danger', 'aria-label': 'Delete ' + f.name, onclick: guard(async () => {
        if (!confirm(`Delete "${f.name}"?`)) return;
        await request('/api/files/' + encodeURIComponent(f.name), {method: 'DELETE'});
        loadFiles();
      })}, '✕')));
  });
});

$('upInput').onchange = async (e) => {
  for (const f of e.target.files) {
    const fd = new FormData();
    fd.append('file', f);
    await guard(() => request('/api/upload', {method: 'POST', body: fd}))();
  }
  e.target.value = '';
  loadFiles();
};

/* ---------- projects ---------- */
const loadProjects = guard(async () => {
  projects = await api('/api/projects');
  const box = $('projects');
  box.replaceChildren();
  if (!projects.length) box.append(el('div', {class: 'muted', style: 'padding:8px 10px'}, 'No projects yet.'));
  const current = activeProjectId();
  projects.forEach((p) => {
    const open = el('button', {class: 'open', type: 'button', onclick: () => openProject(p.id)}, p.name);
    if (p.id === current) open.setAttribute('aria-current', 'true');
    box.appendChild(el('div', {class: 'item' + (p.id === current ? ' active' : '')}, open));
  });
});

const openProject = guard(async (id) => {
  curProject = await api('/api/projects/' + id);
  sid = null; session = null;
  paintProject();
  $('upResult').replaceChildren();
  $('upResult').hidden = true;
  loadSessions();
  loadProjects();
  toggleNav(false);
});

// Re-read the open project (after an upload, delete or chat) without changing views.
const refreshProject = guard(async () => {
  if (!curProject || $('projectView').hidden) return;
  curProject = await api('/api/projects/' + curProject.id);
  paintProject();
});

const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

function paintProject() {
  const p = curProject;
  showView('project');
  $('title').textContent = p.name;
  $('title').title = '';
  $('renameBtn').hidden = true;
  $('sub').textContent = ['Project', plural(p.sessions.length, 'chat'), plural(p.files.length, 'file')].join(' \u00b7 ');
  $('projLink').hidden = true;
  $('projName').textContent = p.name;
  sessionList($('projSessions'), p.sessions, 'No chats in this project yet.');
  paintProjectFiles(p.files);
}

function paintProjectFiles(files) {
  const box = $('projFiles');
  box.replaceChildren();
  if (!files.length) { box.append(el('div', {class: 'muted'}, 'No files yet. Upload some and every chat in this project can use them.')); return; }
  files.forEach((f) => {
    box.append(el('div', {class: 'line'},
      el('span', {class: 'grow', title: f.path}, f.path),
      el('span', {class: 'badge' + (f.chunks ? ' ok' : ' warn')}, f.chunks ? plural(f.chunks, 'chunk') : 'no text'),
      el('button', {class: 'ghost danger', 'aria-label': 'Delete ' + f.path, title: 'Delete file', onclick: () => deleteProjectFile(f.path)}, '✕')));
  });
}

const deleteProjectFile = guard(async (path) => {
  if (!confirm(`Delete "${path}" from this project? Its indexed content is removed too.`)) return;
  await request(`/api/projects/${curProject.id}/files?path=${encodeURIComponent(path)}`, {method: 'DELETE'});
  refreshProject();
});

$('projNewChat').onclick = () => { if (curProject) newSession(curProject.id); };
$('projLink').onclick = () => { if (session && session.project_id) openProject(session.project_id); };

$('projDelBtn').onclick = guard(async () => {
  const p = curProject;
  if (!p) return;
  const msg = `Delete project "${p.name}"?\n\nThis permanently deletes its ${plural(p.sessions.length, 'chat')} with all messages `
    + `and its ${plural(p.files.length, 'indexed file')}. This cannot be undone.`;
  if (!confirm(msg)) return;
  await request('/api/projects/' + p.id, {method: 'DELETE'});
  curProject = null; sid = null; session = null;
  render();
  refreshLists();
  toast('Project deleted.', 'ok');
});

const createProject = guard(async () => {
  const name = $('projNameInput').value.trim();
  if (!name) { toast('Give the project a name.'); return; }
  const r = await api('/api/projects', json('POST', {name}));
  $('newProjDlg').close();
  await openProject(r.id);
});
$('newProjBtn').onclick = () => { $('projNameInput').value = ''; $('newProjDlg').showModal(); $('projNameInput').focus(); };
$('createProj').onclick = () => createProject();
$('projNameInput').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.isComposing) { e.preventDefault(); createProject(); }
});

/* ---------- project files: upload and indexing ---------- */
const RESULT_TEXT = {
  indexed: 'Indexed.', captioned: 'Image described and indexed.', unchanged: 'Already up to date.',
  skipped: 'Skipped: empty, binary or too large.',
  skipped_pdf_no_text: 'Skipped: this PDF has no text layer (scanned?).',
  skipped_pdf_unreadable: 'Skipped: this PDF is encrypted or corrupt.',
  skipped_image_failed: 'Skipped: the vision model could not describe the image (is it pulled?).',
  skipped_image_disabled: 'Skipped: image support is off (VISION_MODEL is empty).',
};

// {tone, text} for one entry of the upload response ({path, result, status?, detail?}).
function resultInfo(x) {
  if (x.result === 'error') {
    return {tone: 'bad', text: x.detail || HINTS[x.status] || `Error ${x.status}`};
  }
  return {tone: x.result.startsWith('skipped') ? 'warn' : 'ok', text: RESULT_TEXT[x.result] || x.result};
}

function setUploadRow(row, tone, text) {
  const badge = row.querySelector('.badge');
  badge.className = 'badge ' + tone;
  badge.textContent = {ok: 'done', warn: 'skipped', bad: 'failed', wait: 'indexing…'}[tone];
  row.querySelector('.note').textContent = text;
}

// One request per file, so each file shows its own progress and outcome
// (a large PDF or an image described by a vision model can take a while).
async function uploadFiles(fileList) {
  const files = [...fileList];
  if (!files.length || !curProject) return;
  const pid = curProject.id;
  const box = $('upResult');
  box.hidden = false;
  box.replaceChildren();
  for (const f of files) {
    const row = el('div', {class: 'line wrap'},
      el('span', {class: 'grow', title: f.name}, f.name), el('span', {class: 'badge'}), el('span', {class: 'note'}));
    box.append(row);
    setUploadRow(row, 'wait', 'Uploading and indexing...');
    const fd = new FormData();
    fd.append('files', f);
    try {
      const r = await api(`/api/projects/${pid}/files`, {method: 'POST', body: fd});
      const info = resultInfo(r.results[0]);
      setUploadRow(row, info.tone, info.text);
    } catch (err) {
      setUploadRow(row, 'bad', err.message);
    }
  }
  if (curProject && curProject.id === pid) refreshProject();
}

$('projUpload').onchange = (e) => {
  const files = [...e.target.files];
  e.target.value = '';
  uploadFiles(files);
};

const dropZone = $('dropZone');
['dragenter', 'dragover'].forEach((t) => dropZone.addEventListener(t, (e) => {
  e.preventDefault();
  dropZone.classList.add('over');
}));
['dragleave', 'drop'].forEach((t) => dropZone.addEventListener(t, (e) => {
  e.preventDefault();
  dropZone.classList.remove('over');
}));
dropZone.addEventListener('drop', (e) => uploadFiles(e.dataTransfer.files));

/* ---------- start ---------- */
// A small banner when Ollama or a needed model is missing; hidden when all is well.
function healthProblem(h) {
  if (!h) return 'Cannot reach the server.';
  if (!h.ollama) return 'Ollama not reachable.';
  if (h.models && !h.models.chat) return `Chat model ${h.model} not installed \u2014 run: ollama pull ${h.model}`;
  if (h.models && !h.models.embed) return `Embedding model ${h.embed_model} not installed \u2014 run: ollama pull ${h.embed_model}`;
  return '';
}

// Never throws: a failed health check must not break the page.
async function checkHealth() {
  let h = null;
  try {
    h = await api('/api/health');
    modelName = h.model || modelName;
    if (h.max_message_chars) { maxChars = h.max_message_chars; updateCount(); }
  } catch (e) { h = null; }
  const msg = healthProblem(h);
  $('health').textContent = msg;
  $('health').hidden = !msg;
}

(async () => {
  await checkHealth();
  render();
  loadSessions();
  loadProjects();
  setInterval(checkHealth, 30000);
})();
