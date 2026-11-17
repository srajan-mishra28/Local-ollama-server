"""Project workspaces: CRUD, sessions inside projects, shared retrieval, schema migration."""
import json
import sqlite3
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app as server  # noqa: E402
import httpx  # noqa: E402
import rag  # noqa: E402
from backend import database, rag_store  # noqa: E402
from test_pdf_images import PNG, FakeCaptioner, make_pdf  # noqa: E402
from test_rag import fake_embedder  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "server.db")
    server.init_db()
    monkeypatch.setattr(
        rag_store, "store",
        rag.RagStore(tmp_path / "rag.db", tmp_path / "copies", fake_embedder, model="fake",
                     captioner=FakeCaptioner("a red logo reading ACME"), vision_model="v"),
    )
    return TestClient(server.app)


def new_project(client, name="p"):
    return client.post("/api/projects", json={"name": name}).json()["id"]


def upload(client, pid, **files):
    r = client.post(f"/api/projects/{pid}/files", files=[("files", (n, d)) for n, d in files.items()])
    assert r.status_code == 200
    return r.json()["results"]


# ---- CRUD ---------------------------------------------------------------------

def test_create_project_by_name_only(client):
    r = client.post("/api/projects", json={"name": "  Thesis  "})
    assert r.status_code == 200 and r.json()["name"] == "Thesis"
    assert client.get("/api/projects").json() == [{"id": r.json()["id"], "name": "Thesis"}]


@pytest.mark.parametrize("name", ["", "   ", "x" * 101])
def test_create_project_rejects_bad_names(client, name):
    assert client.post("/api/projects", json={"name": name}).status_code == 400
    assert client.get("/api/projects").json() == []


def test_get_project_has_sessions_and_files(client):
    pid = new_project(client)
    s1 = client.post(f"/api/projects/{pid}/sessions", json={"title": "one"}).json()["id"]
    s2 = client.post(f"/api/projects/{pid}/sessions").json()["id"]
    upload(client, pid, **{"a.md": b"alpha notes"})
    got = client.get(f"/api/projects/{pid}").json()
    assert got["name"] == "p"
    assert [s["id"] for s in got["sessions"]] == [s2, s1]  # newest first
    assert [s["title"] for s in got["sessions"]] == ["New chat", "one"]
    assert got["files"] == [{"path": "a.md", "chunks": 1}]


def test_unknown_project_is_404_everywhere(client):
    assert client.get("/api/projects/9").status_code == 404
    assert client.delete("/api/projects/9").json() == {"ok": True}  # idempotent like session delete
    assert client.post("/api/projects/9/sessions").status_code == 404
    assert client.post("/api/sessions", json={"project_id": 9}).status_code == 404


# ---- sessions inside projects --------------------------------------------------

def test_sessions_are_listed_under_their_project_not_among_chats(client):
    a = new_project(client, "a")
    b = new_project(client, "b")
    normal = client.post("/api/sessions", json={"title": "plain"}).json()["id"]
    in_a = client.post(f"/api/projects/{a}/sessions").json()["id"]
    in_b = client.post("/api/sessions", json={"title": "via body", "project_id": b}).json()["id"]
    assert [s["id"] for s in client.get("/api/sessions").json()] == [normal]
    assert [s["id"] for s in client.get(f"/api/projects/{a}").json()["sessions"]] == [in_a]
    assert [s["id"] for s in client.get(f"/api/projects/{b}").json()["sessions"]] == [in_b]
    got = client.get(f"/api/sessions/{in_a}").json()
    assert got["project_id"] == a and got["project_name"] == "a"
    assert client.get(f"/api/sessions/{normal}").json()["project_name"] is None


def test_project_session_endpoint_ignores_project_id_in_body(client):
    a, b = new_project(client, "a"), new_project(client, "b")
    sid = client.post(f"/api/projects/{a}/sessions", json={"title": "t", "project_id": b}).json()["id"]
    assert client.get(f"/api/sessions/{sid}").json()["project_id"] == a


# ---- cascade delete ---------------------------------------------------------------

def test_delete_project_cascades_but_leaves_others_alone(client):
    a, b = new_project(client, "a"), new_project(client, "b")
    sa = client.post(f"/api/projects/{a}/sessions").json()["id"]
    sb = client.post(f"/api/projects/{b}/sessions").json()["id"]
    plain = client.post("/api/sessions").json()["id"]
    with database.db() as c:
        for s in (sa, sb, plain):
            c.execute("INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,0)",
                      (s, "user", "hi"))
    upload(client, a, **{"a.md": b"alpha notes"})
    upload(client, b, **{"b.md": b"beta notes"})

    assert client.delete(f"/api/projects/{a}").json() == {"ok": True}

    assert client.get(f"/api/projects/{a}").status_code == 404
    assert client.get(f"/api/sessions/{sa}").status_code == 404
    with database.db() as c:
        assert {r[0] for r in c.execute("SELECT DISTINCT session_id FROM messages")} == {sb, plain}
    store = rag_store.store
    assert not store.has_index(a) and store.list_files(a) == [] and not (store.files_dir / str(a)).exists()
    assert store.has_index(b) and (store.files_dir / str(b) / "b.md").exists()
    assert client.get(f"/api/sessions/{plain}").status_code == 200


# ---- chat: normal chats vs shared project retrieval -------------------------------------

class _FakeResp:
    async def aiter_lines(self):
        yield json.dumps({"message": {"content": "ok"}})

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


@pytest.fixture
def ollama(monkeypatch):
    """Replaces the streaming Ollama client; the list collects each request's JSON body."""
    captured = []

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def stream(self, method, url, json=None):
            captured.append(json)
            return _FakeResp()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    return captured


def system_text(request):
    return "\n".join(m["content"] for m in request["messages"] if m["role"] == "system")


def test_normal_chat_never_retrieves(client, ollama, monkeypatch):
    pid = new_project(client)
    upload(client, pid, **{"a.md": b"banana smoothie recipe"})
    calls = []
    monkeypatch.setattr(rag_store.store, "query", lambda *a, **k: calls.append(a) or [])
    sid = client.post("/api/sessions").json()["id"]
    r = client.post("/api/chat", json={"session_id": sid, "content": "banana smoothie"})
    assert r.status_code == 200
    assert calls == [] and system_text(ollama[0]) == ""
    assert json.loads(r.headers["X-RAG-Sources"]) == []


def test_two_sessions_of_a_project_retrieve_the_same_uploaded_file(client, ollama):
    pid = new_project(client)
    upload(client, pid, **{"recipe.md": b"banana smoothie recipe"})
    s1 = client.post(f"/api/projects/{pid}/sessions").json()["id"]
    s2 = client.post(f"/api/projects/{pid}/sessions").json()["id"]
    for sid in (s1, s2):
        r = client.post("/api/chat", json={"session_id": sid, "content": "banana smoothie"})
        assert json.loads(r.headers["X-RAG-Sources"]) == ["recipe.md"]
    assert "banana smoothie recipe" in system_text(ollama[0])
    assert system_text(ollama[0]) == system_text(ollama[1])


def test_a_session_does_not_see_another_projects_files(client, ollama):
    a, b = new_project(client, "a"), new_project(client, "b")
    upload(client, a, **{"secret.md": b"banana smoothie recipe"})
    sid = client.post(f"/api/projects/{b}/sessions").json()["id"]
    client.post("/api/chat", json={"session_id": sid, "content": "banana smoothie"})
    assert system_text(ollama[0]) == ""


def test_files_uploaded_later_are_used_by_existing_sessions(client, ollama):
    pid = new_project(client)
    sid = client.post(f"/api/projects/{pid}/sessions").json()["id"]
    client.post("/api/chat", json={"session_id": sid, "content": "quantum entanglement"})
    assert system_text(ollama[0]) == ""
    upload(client, pid, **{"q.md": b"quantum entanglement notes"})
    client.post("/api/chat", json={"session_id": sid, "content": "quantum entanglement"})
    assert "quantum entanglement notes" in system_text(ollama[1])


def test_retrieval_query_includes_recent_user_turns(client, ollama, monkeypatch):
    pid = new_project(client)
    upload(client, pid, **{"a.md": b"x y"})
    seen = []
    real = rag_store.store.query
    monkeypatch.setattr(rag_store.store, "query", lambda p, text, k: seen.append(text) or real(p, text, k))
    sid = client.post(f"/api/projects/{pid}/sessions").json()["id"]
    client.post("/api/chat", json={"session_id": sid, "content": "tell me about the bananas"})
    client.post("/api/chat", json={"session_id": sid, "content": "and what about the second one?"})
    assert seen[0] == "tell me about the bananas"
    assert seen[1] == "tell me about the bananas\nand what about the second one?"


def test_excerpts_total_size_is_bounded(client, ollama, monkeypatch):
    from backend.routes import chat
    monkeypatch.setattr(chat, "RAG_MAX_EXCERPT_CHARS", 500)
    pid = new_project(client)
    upload(client, pid, **{f"f{i}.md": (f"banana {i} " * 150).encode() for i in range(4)})  # ~1 chunk each
    sid = client.post(f"/api/projects/{pid}/sessions").json()["id"]
    r = client.post("/api/chat", json={"session_id": sid, "content": "banana"})
    excerpts = system_text(ollama[0]).split("<excerpts>")[1]
    body_chars = sum(len(part.split("\n", 1)[1]) for part in excerpts.split("--- ")[1:])
    assert 0 < body_chars < 500 + 200  # headers and the closing tag are not excerpt text
    assert 1 <= len(json.loads(r.headers["X-RAG-Sources"])) < 4


def test_history_window_drops_old_turns_but_keeps_latest(client, ollama, monkeypatch):
    from backend import prompt
    monkeypatch.setattr(prompt.fit_history, "__defaults__", (1000,))
    sid = client.post("/api/sessions").json()["id"]
    with database.db() as c:
        for i, role in enumerate(["user", "assistant", "user", "assistant"]):
            c.execute("INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
                      (sid, role, f"m{i} " + "x" * 700, i))
    client.post("/api/chat", json={"session_id": sid, "content": "the latest question"})
    sent = [m["content"][:2] for m in ollama[0]["messages"]]
    assert sent[-1] == "th"  # the latest message is last, and always present
    assert sent[:-1] == ["m3"]  # only the newest old turn fits; order preserved
    # everything stays stored and visible
    stored = client.get(f"/api/sessions/{sid}").json()["messages"]
    assert [m["content"][:2] for m in stored[:4]] == ["m0", "m1", "m2", "m3"]


# ---- upload: multi-file partial failures ---------------------------------------------------

def test_multi_upload_reports_each_file_and_keeps_the_good_ones(client):
    pid = new_project(client)
    big = b"a " * (rag.MAX_FILE_BYTES // 2 + 10)
    results = upload(client, pid, **{
        "good.md": b"remember the milk", "big.txt": big, "bin.dat": b"\x00\x01",
        "CON.txt": b"x y", "scan.pdf": make_pdf([""]), "logo.png": PNG, "doc.pdf": make_pdf(["hello pdf"]),
    })
    by = {r["path"]: r for r in results}
    assert [r["path"] for r in results] == ["good.md", "big.txt", "bin.dat", "CON.txt", "scan.pdf", "logo.png", "doc.pdf"]
    assert by["good.md"]["result"] == "indexed"
    assert by["big.txt"]["result"] == "error" and by["big.txt"]["status"] == 413
    assert by["bin.dat"]["result"] == "skipped"
    assert by["CON.txt"]["result"] == "error" and by["CON.txt"]["status"] == 400
    assert by["scan.pdf"]["result"] == "skipped_pdf_no_text"
    assert by["logo.png"]["result"] == "captioned" and by["doc.pdf"]["result"] == "indexed"
    # files without text are recorded (visible, deletable) but have no chunks and no copy
    listed = {f["path"]: f["chunks"] for f in client.get(f"/api/projects/{pid}/files").json()}
    assert set(listed) == {"good.md", "logo.png", "doc.pdf", "bin.dat", "scan.pdf"}
    assert listed["bin.dat"] == listed["scan.pdf"] == 0 and listed["good.md"] >= 1
    assert not (rag_store.store.files_dir / str(pid) / "bin.dat").exists()


def test_upload_is_the_only_way_in_and_indexes_immediately(client):
    pid = new_project(client)
    assert client.get(f"/api/projects/{pid}/rag", params={"q": "milk"}).json() == []
    upload(client, pid, **{"n.md": b"remember the milk"})
    assert client.get(f"/api/projects/{pid}/rag", params={"q": "milk"}).json()[0]["path"] == "n.md"
    assert (rag_store.store.files_dir / str(pid) / "n.md").read_bytes() == b"remember the milk"


# ---- migration from the old folder-based schema -----------------------------------------------

OLD_SCHEMA = """
CREATE TABLE projects (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, path TEXT NOT NULL);
CREATE TABLE sessions (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, project_id INTEGER,
                       context TEXT NOT NULL DEFAULT '[]', created_at REAL NOT NULL);
CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER NOT NULL, role TEXT NOT NULL,
                       content TEXT NOT NULL, created_at REAL NOT NULL);
CREATE INDEX idx_msg_session ON messages(session_id);
"""


@pytest.fixture
def old_db(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(OLD_SCHEMA)
    conn.execute("INSERT INTO projects (name, path) VALUES ('Old folder', 'C:\\\\work\\\\x')")
    conn.executemany(
        "INSERT INTO sessions (title, project_id, context, created_at) VALUES (?,?,?,?)",
        [("in project", 1, '["a.py", "b.py"]', 1.0), ("plain", None, "[]", 2.0),
         ("dangling", 77, '["gone.py"]', 3.0)],  # its project was deleted long ago
    )
    conn.executemany("INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,0)",
                     [(1, "user", "hello"), (1, "assistant", "hi"), (3, "user", "orphan")])
    conn.commit()
    conn.close()
    monkeypatch.setattr(database, "DB_PATH", path)
    return path


def snapshot(path):
    conn = sqlite3.connect(path)
    try:
        return {t: conn.execute(f"SELECT * FROM {t} ORDER BY id").fetchall()
                for t in ("projects", "sessions", "messages")}
    finally:
        conn.close()


def test_migration_keeps_old_data_usable(old_db):
    server.init_db()
    c = TestClient(server.app)
    assert c.get("/api/projects").json() == [{"id": 1, "name": "Old folder"}]
    project = c.get("/api/projects/1").json()
    assert [s["title"] for s in project["sessions"]] == ["in project"]  # keeps its project
    assert project["files"] == []  # old folder projects become empty workspaces
    got = c.get("/api/sessions/1").json()
    assert got["project_name"] == "Old folder" and "context" not in got
    assert [m["content"] for m in got["messages"]] == ["hello", "hi"]
    # a session whose project no longer exists becomes a normal chat, with its messages
    assert sorted(s["title"] for s in c.get("/api/sessions").json()) == ["dangling", "plain"]
    assert c.get("/api/sessions/3").json()["messages"][0]["content"] == "orphan"
    # new projects and sessions work against the old table definitions (path NOT NULL)
    pid = c.post("/api/projects", json={"name": "New"}).json()["id"]
    assert c.post(f"/api/projects/{pid}/sessions").status_code == 200
    assert c.post("/api/sessions", json={"project_id": 1}).status_code == 200


def test_migration_is_idempotent_and_deletes_no_rows(old_db):
    server.init_db()
    once = snapshot(old_db)
    server.init_db()
    assert snapshot(old_db) == once
    assert len(once["projects"]) == 1 and len(once["sessions"]) == 3 and len(once["messages"]) == 3
    # only the dangling session's project link changed; the legacy columns are untouched
    assert once["sessions"][2][2] is None and once["sessions"][0][2] == 1
    assert once["sessions"][0][3] == '["a.py", "b.py"]'
    assert once["projects"][0][2] == "C:\\\\work\\\\x"


def test_migration_adds_missing_project_id_column(tmp_path, monkeypatch):
    path = tmp_path / "older.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        OLD_SCHEMA.replace("project_id INTEGER,\n                       context", "context"))
    conn.execute("INSERT INTO sessions (title, created_at) VALUES ('s', 1)")
    conn.commit()
    conn.close()
    monkeypatch.setattr(database, "DB_PATH", path)
    server.init_db()
    assert [s["title"] for s in TestClient(server.app).get("/api/sessions").json()] == ["s"]


def test_init_db_on_a_fresh_database_is_repeatable(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "fresh.db")
    server.init_db()
    server.init_db()
    c = TestClient(server.app)
    pid = c.post("/api/projects", json={"name": "p"}).json()["id"]
    assert c.get(f"/api/projects/{pid}").json()["sessions"] == []
