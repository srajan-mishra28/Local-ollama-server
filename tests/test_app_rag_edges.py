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
def project(client):
    """A project with a.py and b.py uploaded (and so already indexed)."""
    pid = client.post("/api/projects", json={"name": "p"}).json()["id"]
    client.post(f"/api/projects/{pid}/files", files=[
        ("files", ("a.py", b"def alpha():\n    return 'banana smoothie'\n")),
        ("files", ("b.py", b"def beta():\n    return 'quantum entanglement'\n")),
    ])
    return pid


def upload(client, pid, name, data):
    r = client.post(f"/api/projects/{pid}/files", files={"files": (name, data)})
    assert r.status_code == 200
    return r.json()["results"][0]


# ---- /rag endpoint ---------------------------------------------------------

def test_rag_blank_query_returns_empty(client, project):
    assert client.get(f"/api/projects/{project}/rag", params={"q": "   "}).json() == []


def test_rag_missing_q_is_422(client, project):
    assert client.get(f"/api/projects/{project}/rag").status_code == 422


@pytest.mark.parametrize("k,expected", [(0, 1), (-5, 1), (1, 1), (2, 2), (1000, 2)])
def test_rag_k_is_clamped(client, project, k, expected):
    r = client.get(f"/api/projects/{project}/rag", params={"q": "banana", "k": k})
    assert r.status_code == 200 and len(r.json()) == expected


def test_rag_unknown_project_404(client):
    assert client.get("/api/projects/99/rag", params={"q": "x"}).status_code == 404


def test_rag_embedding_failure_is_502(client, project, monkeypatch):
    def boom(texts):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(rag_store.store, "embedder", boom)
    assert client.get(f"/api/projects/{project}/rag", params={"q": "banana"}).status_code == 502


def test_rag_after_model_change_is_409(client, project, monkeypatch):
    monkeypatch.setattr(rag_store.store, "model", "other-model")
    assert client.get(f"/api/projects/{project}/rag", params={"q": "banana"}).status_code == 409


def test_rag_bad_embedder_response_is_502_not_400(client, project, monkeypatch):
    def bad(texts):
        raise rag.EmbeddingError("garbage")

    monkeypatch.setattr(rag_store.store, "embedder", bad)
    assert client.get(f"/api/projects/{project}/rag", params={"q": "banana"}).status_code == 502
    res = upload(client, project, "n.md", b"x y")
    assert res["result"] == "error" and res["status"] == 502


# ---- files endpoints ---------------------------------------------------------

def test_files_listing_has_chunk_counts_sorted(client, project):
    assert client.get(f"/api/projects/{project}/files").json() == [
        {"path": "a.py", "chunks": 1}, {"path": "b.py", "chunks": 1}]


def test_reupload_unchanged_is_reported(client, project):
    assert upload(client, project, "a.py", b"def alpha():\n    return 'banana smoothie'\n")["result"] == "unchanged"


def test_upload_filename_traversal_is_sanitised(client, project, tmp_path):
    assert upload(client, project, "../../evil.md", b"x y z") == {"path": "evil.md", "result": "indexed"}
    assert not (tmp_path / "evil.md").exists()


def test_upload_replaces_same_name(client, project):
    upload(client, project, "n.md", b"remember the milk")
    upload(client, project, "n.md", b"buy eggs")
    texts = [h["text"] for h in client.get(f"/api/projects/{project}/rag", params={"q": "milk", "k": 10}).json()
             if h["path"] == "n.md"]
    assert texts == ["buy eggs"]


def test_upload_unicode_and_crlf(client, project):
    data = "naïve café\r\nüber ✓\r\n".encode()
    assert upload(client, project, "u.txt", data)["result"] == "indexed"
    hit = client.get(f"/api/projects/{project}/rag", params={"q": "café"}).json()[0]
    assert "café" in hit["text"]


def test_upload_oversize_is_413_entry_and_stores_nothing(client, project):
    big = b"a " * (rag.MAX_FILE_BYTES // 2 + 10)
    res = upload(client, project, "big.txt", big)
    assert res["result"] == "error" and res["status"] == 413 and "too large" in res["detail"]
    assert not (rag_store.store.files_dir / str(project) / "big.txt").exists()
    assert "big.txt" not in [f["path"] for f in rag_store.store.list_files(project)]


def test_upload_exactly_at_cap_is_accepted(client, project):
    assert upload(client, project, "edge.txt", b"a" * rag.MAX_FILE_BYTES)["result"] == "indexed"


@pytest.mark.parametrize("bad", ["n.md:stream", "CON", "x.md.", "COM1.txt"])
def test_upload_windows_name_tricks_are_400_entries(client, project, bad):
    res = upload(client, project, bad, b"x y")
    assert res["result"] == "error" and res["status"] == 400
    assert bad not in [f["path"] for f in rag_store.store.list_files(project)]


def test_upload_binary_stores_no_copy(client, project):
    upload(client, project, "b.bin", b"\x00\x01")
    assert not (rag_store.store.files_dir / str(project) / "b.bin").exists()


def test_upload_requires_files_and_caps_their_number(client, project, monkeypatch):
    assert client.post(f"/api/projects/{project}/files").status_code == 422
    from backend.routes import project_files
    monkeypatch.setattr(project_files, "MAX_UPLOAD_FILES", 2)
    r = client.post(f"/api/projects/{project}/files", files=[("files", (f"{i}.md", b"x y")) for i in range(3)])
    assert r.status_code == 400 and "too many" in r.json()["detail"]


# ---- delete project cleanup -----------------------------------------------

def test_delete_project_removes_index_and_copies(client, project):
    assert (rag_store.store.files_dir / str(project)).exists()
    assert client.delete(f"/api/projects/{project}").status_code == 200
    assert not rag_store.store.has_index(project)
    assert rag_store.store.list_files(project) == []
    assert not (rag_store.store.files_dir / str(project)).exists()


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
    captured = []
    monkeypatch.setattr(httpx, "AsyncClient", _fake_client_factory(captured))
    sid = client.post("/api/sessions", json={"title": "t", "project_id": project}).json()["id"]
    return client, project, sid, captured


def test_chat_injects_retrieved_excerpts(chat_env):
    client, pid, sid, captured = chat_env
    r = client.post("/api/chat", json={"session_id": sid, "content": "banana smoothie"})
    assert r.text == "hi there"
    msgs = captured[0]["messages"]
    systems = [m for m in msgs if m["role"] == "system"]
    assert len(systems) == 1 and "a.py" in systems[0]["content"] and "banana" in systems[0]["content"]
    assert msgs[-1] == {"role": "user", "content": "banana smoothie"}


def test_chat_reports_sources_in_header(chat_env):
    client, _, sid, _ = chat_env
    r = client.post("/api/chat", json={"session_id": sid, "content": "banana smoothie"})
    assert "a.py" in json.loads(r.headers["X-RAG-Sources"])


def test_chat_without_index_has_no_system_message(client, monkeypatch):
    captured = []
    monkeypatch.setattr(httpx, "AsyncClient", _fake_client_factory(captured))
    pid = client.post("/api/projects", json={"name": "empty"}).json()["id"]
    sid = client.post(f"/api/projects/{pid}/sessions").json()["id"]
    r = client.post("/api/chat", json={"session_id": sid, "content": "banana"})
    assert all(m["role"] != "system" for m in captured[0]["messages"])
    assert json.loads(r.headers["X-RAG-Sources"]) == []


def test_chat_falls_back_when_embedding_fails(chat_env, monkeypatch):
    client, pid, sid, captured = chat_env

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
    monkeypatch.setattr(rag_store.store, "embedder", lambda t: [[1.0, 2.0, 3.0]] * len(t))
    # dimension mismatch against stored vectors must not break chat
    assert await server.retrieve_context(project, "banana") == ""


@pytest.mark.anyio
@pytest.mark.parametrize("exc", [
    sqlite3.OperationalError("locked"), OSError("disk"), ValueError("v"),
    rag.ReindexRequired("r"), httpx.ReadTimeout("slow"),
])
async def test_retrieve_context_swallows_broad_failures(client, project, monkeypatch, exc):
    def boom(*a, **kw):
        raise exc

    monkeypatch.setattr(rag_store.store, "query", boom)
    assert await server.retrieve_context(project, "banana") == ""


@pytest.mark.anyio
async def test_retrieve_context_swallows_has_index_db_error(client, project, monkeypatch):
    def boom(*a):
        raise sqlite3.OperationalError("locked")

    monkeypatch.setattr(rag_store.store, "has_index", boom)
    assert await server.retrieve_context(project, "banana") == ""
