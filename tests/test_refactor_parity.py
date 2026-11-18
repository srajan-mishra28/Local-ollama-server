"""Parity tests for the app.py -> backend/ + static/ split.

Expected behaviour is taken from `git show HEAD:app.py` (the pre-refactor
single file).
"""
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import app as server  # noqa: E402
from backend import config, database  # noqa: E402
from backend.routes import chat, project_files, projects, sessions  # noqa: E402
from backend.routes import files as files_routes  # noqa: E402


def _head_app():
    try:
        r = subprocess.run(["git", "show", "HEAD:app.py"], cwd=ROOT, capture_output=True,
                           text=True, encoding="utf-8", timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "server.db")
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setattr(files_routes, "UPLOADS", uploads)
    server.init_db()
    return TestClient(server.app)


@pytest.fixture
def uploads(client):
    return files_routes.UPLOADS


# ---------------------------------------------------------------- imports / routes

def test_reexports_are_importable():
    from app import app, retrieve_context  # noqa: F401
    assert callable(retrieve_context)


def test_no_duplicate_routes():
    current = Counter()
    # Newer FastAPI wraps include_router() results in opaque objects, so collect
    # from the routers that app.py includes plus app's own routes.
    routers = [mod.router for mod in (sessions, projects, project_files, files_routes, chat)]
    routes = [r for rt in routers for r in rt.routes] + [
        r for r in server.app.routes if isinstance(r, APIRoute)]
    for r in routes:
        for m in r.methods:
            current[(m.lower(), r.path)] += 1
    assert len(current) >= 15
    dupes = {k: v for k, v in current.items() if v > 1}
    assert not dupes, f"duplicate routes: {dupes}"


def test_removed_folder_endpoints_are_gone():
    paths = {r.path for r in server.app.routes if isinstance(r, APIRoute)}
    for p in ("/api/projects/{pid}/grep", "/api/projects/{pid}/index", "/api/projects/{pid}/index/file",
              "/api/projects/{pid}/upload", "/api/sessions/{sid}/context", "/api/sessions/{sid}/project"):
        assert p not in paths, p


# ---------------------------------------------------------------- UI / static

def test_index_serves_html_referencing_static_assets(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert '<link rel="stylesheet" href="/static/style.css">' in r.text
    assert '<script src="/static/app.js"></script>' in r.text
    assert "<style>" not in r.text and "<script>" not in r.text  # nothing left inline


@pytest.mark.parametrize("path,ctype", [
    ("/static/style.css", "text/css"),
    ("/static/app.js", "javascript"),
    ("/static/index.html", "text/html"),
])
def test_static_assets_served(client, path, ctype):
    r = client.get(path)
    assert r.status_code == 200
    assert ctype in r.headers["content-type"]
    assert len(r.content) > 0


def test_static_missing_is_404(client):
    assert client.get("/static/nope.js").status_code == 404


# ---------------------------------------------------------------- health

def test_health_shape(client):
    body = client.get("/api/health").json()
    assert body["ok"] is True and body["model"] == config.MODEL  # fields older clients rely on


# ---------------------------------------------------------------- sessions

def test_session_lifecycle(client):
    assert client.get("/api/sessions").json() == []
    sid = client.post("/api/sessions", json={}).json()["id"]
    got = client.get(f"/api/sessions/{sid}").json()
    assert got["title"] == "New chat"
    assert got["project_id"] is None
    assert got["messages"] == []
    assert set(got) == {"id", "title", "project_id", "project_name", "created_at", "messages"}

    assert client.delete(f"/api/sessions/{sid}").json() == {"ok": True}
    r = client.get(f"/api/sessions/{sid}")
    assert r.status_code == 404
    assert r.json() == {"detail": "no such session"}


def test_list_sessions_is_normal_chats_only_newest_first(client):
    pid = client.post("/api/projects", json={"name": "proj"}).json()["id"]
    a = client.post("/api/sessions", json={"title": "first"}).json()["id"]
    client.post("/api/sessions", json={"title": "in project", "project_id": pid})
    b = client.post("/api/sessions", json={"title": "second"}).json()["id"]
    rows = client.get("/api/sessions").json()
    assert [r["id"] for r in rows] == [b, a]
    assert all(r["project_id"] is None for r in rows)


def test_list_sessions_created_at_tie_breaks_by_id_desc(client):
    ids = [client.post("/api/sessions", json={}).json()["id"] for _ in range(3)]
    with database.db() as c:
        c.execute("UPDATE sessions SET created_at = 1")
    assert [r["id"] for r in client.get("/api/sessions").json()] == ids[::-1]


def test_delete_session_removes_its_messages(client):
    sid = client.post("/api/sessions", json={}).json()["id"]
    with database.db() as c:
        c.execute("INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,0)",
                  (sid, "user", "hi"))
    assert client.get(f"/api/sessions/{sid}").json()["messages"] == [{"role": "user", "content": "hi"}]
    client.delete(f"/api/sessions/{sid}")
    with database.db() as c:
        assert c.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0


def test_delete_unknown_session_is_ok(client):
    assert client.delete("/api/sessions/999").json() == {"ok": True}


# ---------------------------------------------------------------- projects

def test_project_crud(client):
    assert client.get("/api/projects").json() == []
    pid = client.post("/api/projects", json={"name": "zeta"}).json()["id"]
    client.post("/api/projects", json={"name": "alpha"})
    rows = client.get("/api/projects").json()
    assert [p["name"] for p in rows] == ["alpha", "zeta"]  # ORDER BY name
    assert set(rows[0]) == {"id", "name"}
    assert client.delete(f"/api/projects/{pid}").json() == {"ok": True}
    assert [p["name"] for p in client.get("/api/projects").json()] == ["alpha"]


def test_create_project_requires_name(client):
    assert client.post("/api/projects", json={}).status_code == 422


# ---------------------------------------------------------------- files

@pytest.mark.parametrize("name,expected", [
    ("a.txt", "a.txt"),
    ("../../etc/passwd", "passwd"),
    ("a..b.txt", "a_b.txt"),
    ("dir/sub/file.bin", "file.bin"),
])
def test_safe_name(name, expected):
    assert files_routes.safe_name(name) == expected


def test_upload_list_download_delete(client, uploads):
    r = client.post("/api/upload", files={"file": ("hello.txt", b"hi there")}, data={"note": "n1"})
    assert r.json() == {"name": "hello.txt", "size": 8, "note": "n1"}
    assert (uploads / "hello.txt").read_bytes() == b"hi there"

    listing = client.get("/api/files").json()
    assert len(listing) == 1 and set(listing[0]) == {"name", "size", "mtime"}
    assert listing[0]["name"] == "hello.txt" and listing[0]["size"] == 8

    d = client.get("/api/files/hello.txt")
    assert d.status_code == 200 and d.content == b"hi there"
    assert "hello.txt" in d.headers["content-disposition"]

    assert client.delete("/api/files/hello.txt").json() == {"ok": True}
    assert client.get("/api/files/hello.txt").status_code == 404
    assert client.get("/api/files").json() == []


def test_upload_note_defaults_to_empty(client):
    r = client.post("/api/upload", files={"file": ("a.txt", b"x")})
    assert r.json()["note"] == ""


def test_upload_name_collisions_get_numeric_suffix(client, uploads):
    names = [client.post("/api/upload", files={"file": ("a.txt", str(i).encode())}).json()["name"]
             for i in range(3)]
    assert names == ["a.txt", "a_1.txt", "a_2.txt"]
    assert (uploads / "a.txt").read_bytes() == b"0"  # earlier upload not overwritten


def test_upload_traversal_name_stays_inside_uploads(client, uploads, tmp_path):
    r = client.post("/api/upload", files={"file": ("../evil.txt", b"x")})
    assert r.json()["name"] == "evil.txt"
    assert (uploads / "evil.txt").exists() and not (tmp_path / "evil.txt").exists()


def test_upload_without_file_is_422(client):
    assert client.post("/api/upload").status_code == 422


def test_download_and_delete_traversal_are_confined(client, uploads, tmp_path):
    (tmp_path / "secret.txt").write_text("s")
    assert client.get("/api/files/..%2Fsecret.txt").status_code == 404
    client.delete("/api/files/..%2Fsecret.txt")
    assert (tmp_path / "secret.txt").exists()


def test_delete_missing_file_is_ok_and_directory_not_listed(client, uploads):
    assert client.delete("/api/files/none.txt").json() == {"ok": True}
    (uploads / "subdir").mkdir()
    assert client.get("/api/files").json() == []
    assert client.get("/api/files/subdir").status_code == 404


def test_list_files_newest_first(client, uploads):
    (uploads / "old.txt").write_text("1")
    (uploads / "new.txt").write_text("2")
    os.utime(uploads / "old.txt", (1000, 1000))
    os.utime(uploads / "new.txt", (2000, 2000))
    assert [f["name"] for f in client.get("/api/files").json()] == ["new.txt", "old.txt"]
