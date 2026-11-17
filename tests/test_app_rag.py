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
    """A project workspace holding two uploaded files, a.py and b.py."""
    pid = client.post("/api/projects", json={"name": "p"}).json()["id"]
    r = client.post(f"/api/projects/{pid}/files", files=[
        ("files", ("a.py", b"def alpha():\n    return 'banana smoothie'\n")),
        ("files", ("b.py", b"def beta():\n    return 'quantum entanglement'\n")),
    ])
    assert [x["result"] for x in r.json()["results"]] == ["indexed", "indexed"]
    return pid


def test_upload_query_replace_delete(client, project):
    pid = project
    hits = client.get(f"/api/projects/{pid}/rag", params={"q": "banana"}).json()
    assert hits[0]["path"] == "a.py"

    r = client.post(f"/api/projects/{pid}/files", files={"files": ("a.py", b"def alpha():\n    return 'cherry'\n")})
    assert r.json()["results"] == [{"path": "a.py", "result": "indexed"}]
    assert all("banana" not in h["text"] for h in client.get(f"/api/projects/{pid}/rag", params={"q": "banana"}).json())

    assert client.delete(f"/api/projects/{pid}/files", params={"path": "a.py"}).json() == {"deleted": True}
    assert [f["path"] for f in client.get(f"/api/projects/{pid}/files").json()] == ["b.py"]


def test_delete_file_and_bad_path(client, project):
    pid = project
    assert client.delete(f"/api/projects/{pid}/files", params={"path": "zzz"}).json() == {"deleted": False}
    assert client.delete(f"/api/projects/{pid}/files", params={"path": "../x"}).status_code == 400


def test_upload_binary_is_skipped(client, project):
    r = client.post(f"/api/projects/{project}/files", files={"files": ("b.bin", b"\x00\x01")})
    assert r.json()["results"] == [{"path": "b.bin", "result": "skipped"}]


def test_unknown_project_404(client):
    assert client.post("/api/projects/99/files", files={"files": ("a.md", b"x")}).status_code == 404
    assert client.get("/api/projects/99/files").status_code == 404
    assert client.get("/api/projects/99").status_code == 404


def test_embedding_failure_is_a_502_entry_and_does_not_stop_the_batch(client, project, monkeypatch):
    calls = []

    def flaky(texts):
        calls.append(texts)
        if len(calls) == 1:
            raise httpx.ConnectError("down")
        return fake_embedder(texts)

    monkeypatch.setattr(rag_store.store, "embedder", flaky)
    r = client.post(f"/api/projects/{project}/files", files=[
        ("files", ("n.md", b"x y")), ("files", ("m.md", b"p q")),
    ])
    assert r.status_code == 200
    first, second = r.json()["results"]
    assert first["result"] == "error" and first["status"] == 502 and "Ollama" in first["detail"]
    assert second == {"path": "m.md", "result": "indexed"}


@pytest.mark.anyio
async def test_retrieve_context(client, project, monkeypatch):
    pid = project
    empty = client.post("/api/projects", json={"name": "empty"}).json()["id"]
    assert await server.retrieve_context(empty, "banana") == ""  # nothing indexed yet
    assert "banana" in await server.retrieve_context(pid, "banana")
    assert await server.retrieve_context(None, "banana") == ""

    def boom(texts):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(rag_store.store, "embedder", boom)
    assert await server.retrieve_context(pid, "banana") == ""  # degrades, no exception
