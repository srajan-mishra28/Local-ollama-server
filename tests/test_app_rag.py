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
def project(client, tmp_path):
    root = tmp_path / "proj"
    (root / "node_modules").mkdir(parents=True)
    (root / "a.py").write_text("def alpha():\n    return 'banana smoothie'\n")
    (root / "node_modules" / "x.js").write_text("ignored")
    pid = client.post("/api/projects", json={"name": "p", "path": str(root)}).json()["id"]
    return pid, root


def test_index_query_update_delete(client, project):
    pid, root = project
    assert client.post(f"/api/projects/{pid}/index").json()["indexed"] == 1
    hits = client.get(f"/api/projects/{pid}/rag", params={"q": "banana"}).json()
    assert hits[0]["path"] == "a.py"

    (root / "a.py").write_text("def alpha():\n    return 'cherry'\n")
    assert client.post(f"/api/projects/{pid}/index/file", params={"path": "a.py"}).json() == {"result": "indexed"}

    (root / "a.py").unlink()
    assert client.post(f"/api/projects/{pid}/index").json()["deleted"] == 1
    assert client.get(f"/api/projects/{pid}/index").json() == []


def test_unindex_file_and_bad_path(client, project):
    pid, _ = project
    client.post(f"/api/projects/{pid}/index")
    assert client.delete(f"/api/projects/{pid}/index/file", params={"path": "a.py"}).json() == {"deleted": True}
    assert client.delete(f"/api/projects/{pid}/index/file", params={"path": "../x"}).status_code == 400


def test_upload_to_project(client, project):
    pid, _ = project
    r = client.post(f"/api/projects/{pid}/upload", files={"file": ("n.md", b"remember the milk")})
    assert r.json() == {"path": "n.md", "result": "indexed"}
    assert client.get(f"/api/projects/{pid}/rag", params={"q": "milk"}).json()[0]["path"] == "n.md"
    r = client.post(f"/api/projects/{pid}/upload", files={"file": ("b.bin", b"\x00\x01")})
    assert r.json()["result"] == "skipped"


def test_unknown_project_404(client):
    assert client.post("/api/projects/99/index").status_code == 404


def test_embedding_failure_is_502(client, project, monkeypatch):
    pid, _ = project

    def boom(texts):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(rag_store.store, "embedder", boom)
    assert client.post(f"/api/projects/{pid}/upload", files={"file": ("n.md", b"x y")}).status_code == 502


@pytest.mark.anyio
async def test_retrieve_context(client, project, monkeypatch):
    pid, _ = project
    assert await server.retrieve_context(pid, "banana") == ""  # nothing indexed yet
    client.post(f"/api/projects/{pid}/index")
    assert "banana" in await server.retrieve_context(pid, "banana")
    assert await server.retrieve_context(None, "banana") == ""

    def boom(texts):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(rag_store.store, "embedder", boom)
    assert await server.retrieve_context(pid, "banana") == ""  # degrades, no exception
