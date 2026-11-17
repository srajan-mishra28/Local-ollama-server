import json
import sqlite3
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app as server  # noqa: E402
import httpx  # noqa: E402
from backend import database, rag_store  # noqa: E402
from backend.routes import rag_index  # noqa: E402
import rag  # noqa: E402
from test_rag import fake_embedder  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "server.db")
    server.init_db()
    monkeypatch.setattr(
        rag_store, "store",
        rag.RagStore(tmp_path / "rag.db", tmp_path / "copies", fake_embedder, model="fake"),
    )
    return TestClient(server.app)


@pytest.fixture
def project(client, tmp_path):
    root = tmp_path / "proj"
    (root / "node_modules").mkdir(parents=True)
    (root / "sub").mkdir()
    (root / "a.py").write_text("def alpha():\n    return 'banana smoothie'\n")
    (root / "sub" / "b.py").write_text("def beta():\n    return 'quantum entanglement'\n")
    (root / "node_modules" / "x.js").write_text("ignored")
    (root / "notes.bin").write_bytes(b"\x00\x01")
    pid = client.post("/api/projects", json={"name": "p", "path": str(root)}).json()["id"]
    return pid, root


# ---- scan_project ----------------------------------------------------------

def test_scan_project_filters_and_posix_paths(project):
    _, root = project
    assert server.scan_project(root) == ["a.py", "sub/b.py"]
    assert server.scan_project(root, q="SUB") == ["sub/b.py"]
    assert len(server.scan_project(root, limit=1)) == 1


# ---- /rag endpoint ---------------------------------------------------------

def test_rag_blank_query_returns_empty(client, project):
    pid, _ = project
    client.post(f"/api/projects/{pid}/index")
    assert client.get(f"/api/projects/{pid}/rag", params={"q": "   "}).json() == []


def test_rag_missing_q_is_422(client, project):
    pid, _ = project
    assert client.get(f"/api/projects/{pid}/rag").status_code == 422


@pytest.mark.parametrize("k,expected", [(0, 1), (-5, 1), (1, 1), (2, 2), (1000, 2)])
def test_rag_k_is_clamped(client, project, k, expected):
    pid, _ = project
    client.post(f"/api/projects/{pid}/index")
    r = client.get(f"/api/projects/{pid}/rag", params={"q": "banana", "k": k})
    assert r.status_code == 200 and len(r.json()) == expected


def test_rag_unknown_project_404(client):
    assert client.get("/api/projects/99/rag", params={"q": "x"}).status_code == 404


def test_rag_embedding_failure_is_502(client, project, monkeypatch):
    pid, _ = project
    client.post(f"/api/projects/{pid}/index")

    def boom(texts):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(rag_store.store, "embedder", boom)
    assert client.get(f"/api/projects/{pid}/rag", params={"q": "banana"}).status_code == 502


# ---- index endpoints -------------------------------------------------------

def test_index_status_unknown_project_404(client):
    assert client.get("/api/projects/99/index").status_code == 404


def test_index_skips_ignored_dirs_and_binary_ext(client, project):
    pid, _ = project
    stats = client.post(f"/api/projects/{pid}/index").json()
    assert stats["indexed"] == 2
    paths = [f["path"] for f in client.get(f"/api/projects/{pid}/index").json()]
    assert paths == ["a.py", "sub/b.py"]


def test_reindex_is_incremental(client, project):
    pid, root = project
    client.post(f"/api/projects/{pid}/index")
    assert client.post(f"/api/projects/{pid}/index").json()["unchanged"] == 2
    (root / "a.py").write_text("def alpha():\n    return 'cherry'\n")
    stats = client.post(f"/api/projects/{pid}/index").json()
    assert stats["indexed"] == 1 and stats["unchanged"] == 1


def test_index_fails_fast_with_502_when_ollama_down(client, project, monkeypatch):
    pid, _ = project
    calls = []

    def boom(texts):
        calls.append(texts)
        raise httpx.ConnectError("down")

    monkeypatch.setattr(rag_store.store, "embedder", boom)
    r = client.post(f"/api/projects/{pid}/index")
    assert r.status_code == 502 and len(calls) == 1


def test_index_does_not_prune_when_scan_hit_cap(client, project, monkeypatch):
    pid, _ = project
    client.post(f"/api/projects/{pid}/index")
    monkeypatch.setattr(rag_index, "RAG_SCAN_LIMIT", 1)
    stats = client.post(f"/api/projects/{pid}/index").json()
    assert stats["deleted"] == 0
    assert len(client.get(f"/api/projects/{pid}/index").json()) == 2


@pytest.mark.parametrize("bad", ["../x", "a/../../x", "/etc/passwd", ""])
def test_index_file_bad_path_400(client, project, bad):
    pid, _ = project
    assert client.post(f"/api/projects/{pid}/index/file", params={"path": bad}).status_code == 400


def test_index_file_missing_returns_missing(client, project):
    pid, _ = project
    r = client.post(f"/api/projects/{pid}/index/file", params={"path": "ghost.py"})
    assert r.json() == {"result": "missing"}


def test_index_file_deleted_on_disk(client, project):
    pid, root = project
    client.post(f"/api/projects/{pid}/index")
    (root / "a.py").unlink()
    r = client.post(f"/api/projects/{pid}/index/file", params={"path": "a.py"})
    assert r.json() == {"result": "deleted"}


def test_index_file_embedding_failure_502(client, project, monkeypatch):
    pid, _ = project

    def boom(texts):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(rag_store.store, "embedder", boom)
    r = client.post(f"/api/projects/{pid}/index/file", params={"path": "a.py"})
    assert r.status_code == 502


def test_unindex_unknown_file_false(client, project):
    pid, _ = project
    assert client.delete(f"/api/projects/{pid}/index/file", params={"path": "zzz"}).json() == {"deleted": False}


# ---- upload ----------------------------------------------------------------

def test_upload_explicit_path_form(client, project):
    pid, _ = project
    r = client.post(f"/api/projects/{pid}/upload", data={"path": "docs/readme.md"},
                    files={"file": ("ignored.md", b"hello docs")})
    assert r.json() == {"path": "docs/readme.md", "result": "indexed"}
    assert (rag_store.store.files_dir / str(pid) / "docs" / "readme.md").exists()


@pytest.mark.parametrize("bad", ["../x.md", "/abs.md", "a/../../b.md"])
def test_upload_path_traversal_400(client, project, bad, tmp_path):
    pid, _ = project
    r = client.post(f"/api/projects/{pid}/upload", data={"path": bad}, files={"file": ("f.md", b"x y")})
    assert r.status_code == 400
    assert not (tmp_path / "x.md").exists()


def test_upload_filename_traversal_is_sanitised(client, project):
    pid, _ = project
    r = client.post(f"/api/projects/{pid}/upload", files={"file": ("../../evil.md", b"x y z")})
    assert r.status_code == 200 and r.json()["path"] == "evil.md"


def test_upload_replaces_same_path(client, project):
    pid, _ = project
    client.post(f"/api/projects/{pid}/upload", files={"file": ("n.md", b"remember the milk")})
    client.post(f"/api/projects/{pid}/upload", files={"file": ("n.md", b"buy eggs")})
    texts = [h["text"] for h in client.get(f"/api/projects/{pid}/rag", params={"q": "milk", "k": 10}).json()]
    assert texts == ["buy eggs"]


def test_upload_survives_project_reindex(client, project):
    pid, _ = project
    client.post(f"/api/projects/{pid}/upload", files={"file": ("n.md", b"remember the milk")})
    stats = client.post(f"/api/projects/{pid}/index").json()
    assert stats["deleted"] == 0
    paths = {f["path"] for f in client.get(f"/api/projects/{pid}/index").json()}
    assert "n.md" in paths


def test_upload_unknown_project_404(client):
    assert client.post("/api/projects/99/upload", files={"file": ("n.md", b"x")}).status_code == 404


def test_upload_unicode_and_crlf(client, project):
    pid, _ = project
    data = "naïve café\r\nüber ✓\r\n".encode()
    assert client.post(f"/api/projects/{pid}/upload", files={"file": ("u.txt", data)}).json()["result"] == "indexed"
    hit = client.get(f"/api/projects/{pid}/rag", params={"q": "café"}).json()[0]
    assert "café" in hit["text"]


def test_upload_oversize_is_413_and_stores_nothing(client, project):
    pid, _ = project
    big = b"a " * (rag.MAX_FILE_BYTES // 2 + 10)
    r = client.post(f"/api/projects/{pid}/upload", files={"file": ("big.txt", big)})
    assert r.status_code == 413
    assert not (rag_store.store.files_dir / str(pid) / "big.txt").exists()
    assert rag_store.store.list_files(pid) == []


def test_upload_exactly_at_cap_is_accepted(client, project):
    pid, _ = project
    data = b"a" * rag.MAX_FILE_BYTES
    r = client.post(f"/api/projects/{pid}/upload", files={"file": ("edge.txt", data)})
    assert r.status_code == 200


@pytest.mark.parametrize("bad", ["C:/Windows/x.md", "C:foo.md", "n.md:stream", "..\\x.md", "CON", "x.md."])
def test_upload_windows_path_tricks_400(client, project, bad):
    pid, _ = project
    r = client.post(f"/api/projects/{pid}/upload", data={"path": bad}, files={"file": ("f.md", b"x y")})
    assert r.status_code == 400
    assert not (rag_store.store.files_dir / str(pid)).exists() or not any((rag_store.store.files_dir / str(pid)).rglob("*"))


def test_upload_binary_stores_no_copy(client, project):
    pid, _ = project
    client.post(f"/api/projects/{pid}/upload", files={"file": ("b.bin", b"\x00\x01")})
    assert not (rag_store.store.files_dir / str(pid) / "b.bin").exists()


def test_upload_not_clobbered_by_index_of_same_path(client, project):
    pid, root = project
    (root / "n.md").write_text("from disk")
    client.post(f"/api/projects/{pid}/upload", files={"file": ("n.md", b"remember the milk")})
    stats = client.post(f"/api/projects/{pid}/index").json()
    assert stats["shadowed"] == 1
    assert client.post(f"/api/projects/{pid}/index/file", params={"path": "n.md"}).json() == {"result": "shadowed"}
    hit = client.get(f"/api/projects/{pid}/rag", params={"q": "milk", "k": 10}).json()
    assert "remember the milk" in [h["text"] for h in hit if h["path"] == "n.md"]


# ---- delete project cleanup -----------------------------------------------

def test_delete_project_removes_index_and_copies(client, project):
    pid, _ = project
    client.post(f"/api/projects/{pid}/index")
    client.post(f"/api/projects/{pid}/upload", files={"file": ("n.md", b"hello there")})
    assert (rag_store.store.files_dir / str(pid)).exists()
    assert client.delete(f"/api/projects/{pid}").status_code == 200
    assert not rag_store.store.has_index(pid)
    assert rag_store.store.list_files(pid) == []
    assert not (rag_store.store.files_dir / str(pid)).exists()


# ---- chat integration ------------------------------------------------------

class _FakeResp:
    async def aiter_lines(self):
        for piece in ("hi", " there"):
            yield json.dumps({"message": {"content": piece}})

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def _fake_client_factory(captured):
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

    return FakeClient


@pytest.fixture
def chat_env(client, project, monkeypatch):
    pid, _ = project
    captured = []
    monkeypatch.setattr(httpx, "AsyncClient", _fake_client_factory(captured))
    sid = client.post("/api/sessions", json={"title": "t", "project_id": pid}).json()["id"]
    return client, pid, sid, captured


def test_chat_injects_retrieved_excerpts(chat_env):
    client, pid, sid, captured = chat_env
    client.post(f"/api/projects/{pid}/index")
    r = client.post("/api/chat", json={"session_id": sid, "content": "banana smoothie"})
    assert r.text == "hi there"
    msgs = captured[0]["messages"]
    systems = [m for m in msgs if m["role"] == "system"]
    assert len(systems) == 1 and "a.py" in systems[0]["content"] and "banana" in systems[0]["content"]
    assert msgs[-1] == {"role": "user", "content": "banana smoothie"}


def test_chat_without_index_has_no_system_message(chat_env):
    client, _, sid, captured = chat_env
    client.post("/api/chat", json={"session_id": sid, "content": "banana"})
    assert all(m["role"] != "system" for m in captured[0]["messages"])


def test_chat_falls_back_when_embedding_fails(chat_env, monkeypatch):
    client, pid, sid, captured = chat_env
    client.post(f"/api/projects/{pid}/index")

    def boom(texts):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(rag_store.store, "embedder", boom)
    r = client.post("/api/chat", json={"session_id": sid, "content": "banana"})
    assert r.status_code == 200 and r.text == "hi there"
    assert all(m["role"] != "system" for m in captured[0]["messages"])


def test_chat_without_project_skips_retrieval(client, monkeypatch):
    captured = []
    monkeypatch.setattr(httpx, "AsyncClient", _fake_client_factory(captured))
    sid = client.post("/api/sessions", json={"title": "t"}).json()["id"]
    assert client.post("/api/chat", json={"session_id": sid, "content": "hello"}).text == "hi there"
    assert [m["role"] for m in captured[0]["messages"]] == ["user"]


def test_chat_persists_assistant_reply(chat_env):
    client, _, sid, _ = chat_env
    client.post("/api/chat", json={"session_id": sid, "content": "hello"})
    msgs = client.get(f"/api/sessions/{sid}").json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["content"] == "hi there"


@pytest.mark.anyio
async def test_retrieve_context_swallows_bad_shape(client, project, monkeypatch):
    pid, _ = project
    client.post(f"/api/projects/{pid}/index")
    monkeypatch.setattr(rag_store.store, "embedder", lambda t: [[1.0, 2.0, 3.0]] * len(t))
    # dimension mismatch against stored vectors must not break chat
    assert await server.retrieve_context(pid, "banana") == ""


# ---- scan filters via /index/file ------------------------------------------

@pytest.mark.parametrize("rel", [".env", ".git/config", "id_rsa"])
def test_index_file_ignores_sensitive_paths(client, project, rel):
    pid, root = project
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text("SECRET=1")
    r = client.post(f"/api/projects/{pid}/index/file", params={"path": rel})
    assert r.json() == {"result": "missing"}
    assert client.get(f"/api/projects/{pid}/index").json() == []


@pytest.mark.parametrize("bad", ["C:/Windows/x.py", "x.py:stream", "con.py"])
def test_index_and_unindex_file_windows_tricks_400(client, project, bad):
    pid, _ = project
    assert client.post(f"/api/projects/{pid}/index/file", params={"path": bad}).status_code == 400
    assert client.delete(f"/api/projects/{pid}/index/file", params={"path": bad}).status_code == 400


# ---- model change / error mapping ------------------------------------------

def test_rag_after_model_change_is_409(client, project, monkeypatch):
    pid, _ = project
    client.post(f"/api/projects/{pid}/index")
    monkeypatch.setattr(rag_store.store, "model", "other-model")
    assert client.get(f"/api/projects/{pid}/rag", params={"q": "banana"}).status_code == 409


def test_rag_bad_embedder_response_is_502_not_400(client, project, monkeypatch):
    pid, _ = project
    client.post(f"/api/projects/{pid}/index")

    def bad(texts):
        raise rag.EmbeddingError("garbage")

    monkeypatch.setattr(rag_store.store, "embedder", bad)
    assert client.get(f"/api/projects/{pid}/rag", params={"q": "banana"}).status_code == 502
    r = client.post(f"/api/projects/{pid}/upload", files={"file": ("n.md", b"x y")})
    assert r.status_code == 502


@pytest.mark.anyio
@pytest.mark.parametrize("exc", [
    sqlite3.OperationalError("locked"), OSError("disk"), ValueError("v"),
    rag.ReindexRequired("r"), httpx.ReadTimeout("slow"),
])
async def test_retrieve_context_swallows_broad_failures(client, project, monkeypatch, exc):
    pid, _ = project
    client.post(f"/api/projects/{pid}/index")

    def boom(*a, **kw):
        raise exc

    monkeypatch.setattr(rag_store.store, "query", boom)
    assert await server.retrieve_context(pid, "banana") == ""


@pytest.mark.anyio
async def test_retrieve_context_swallows_has_index_db_error(client, project, monkeypatch):
    pid, _ = project

    def boom(*a):
        raise sqlite3.OperationalError("locked")

    monkeypatch.setattr(rag_store.store, "has_index", boom)
    assert await server.retrieve_context(pid, "banana") == ""
