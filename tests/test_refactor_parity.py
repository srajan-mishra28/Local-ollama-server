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
from backend.routes import chat, projects, rag_index, sessions  # noqa: E402
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
    from app import app, retrieve_context, scan_project  # noqa: F401
    assert callable(retrieve_context) and callable(scan_project)


def test_original_routes_all_still_registered_without_duplicates():
    src = _head_app()
    if src is None:
        pytest.skip("git history not available")
    original = set(re.findall(r'^@app\.(get|post|put|delete)\("([^"]+)"', src, re.M))
    assert len(original) >= 15
    current = Counter()
    # Newer FastAPI wraps include_router() results in opaque objects, so collect
    # from the routers that app.py includes plus app's own routes.
    routers = [mod.router for mod in (sessions, projects, rag_index, files_routes, chat)]
    routes = [r for rt in routers for r in rt.routes] + [
        r for r in server.app.routes if isinstance(r, APIRoute)]
    for r in routes:
        for m in r.methods:
            current[(m.lower(), r.path)] += 1
    missing = original - set(current)
    assert not missing, f"routes lost in refactor: {missing}"
    dupes = {k: v for k, v in current.items() if v > 1}
    assert not dupes, f"duplicate routes: {dupes}"


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
    assert client.get("/api/health").json() == {"ok": True, "model": config.MODEL}


# ---------------------------------------------------------------- sessions

def test_session_lifecycle(client):
    assert client.get("/api/sessions").json() == []
    sid = client.post("/api/sessions", json={}).json()["id"]
    got = client.get(f"/api/sessions/{sid}").json()
    assert got["title"] == "New chat"
    assert got["project_id"] is None
    assert got["context"] == []
    assert got["messages"] == []
    assert set(got) == {"id", "title", "project_id", "context", "created_at", "messages"}

    r = client.put(f"/api/sessions/{sid}/context", json={"files": ["a.py", "b/c.py"]})
    assert r.json() == {"ok": True, "files": ["a.py", "b/c.py"]}
    assert client.get(f"/api/sessions/{sid}").json()["context"] == ["a.py", "b/c.py"]

    assert client.delete(f"/api/sessions/{sid}").json() == {"ok": True}
    r = client.get(f"/api/sessions/{sid}")
    assert r.status_code == 404
    assert r.json() == {"detail": "no such session"}


def test_list_sessions_newest_first_with_project_name(client, tmp_path):
    pid = client.post("/api/projects", json={"name": "proj", "path": str(tmp_path)}).json()["id"]
    a = client.post("/api/sessions", json={"title": "first"}).json()["id"]
    b = client.post("/api/sessions", json={"title": "second", "project_id": pid}).json()["id"]
    rows = client.get("/api/sessions").json()
    assert [r["id"] for r in rows] == [b, a]
    assert rows[0]["project_name"] == "proj"
    assert rows[1]["project_name"] is None


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


def test_set_session_project_binds_and_unbinds(client, tmp_path):
    pid = client.post("/api/projects", json={"name": "p", "path": str(tmp_path)}).json()["id"]
    sid = client.post("/api/sessions", json={}).json()["id"]
    assert client.put(f"/api/sessions/{sid}/project", params={"pid": str(pid)}).json() == {"ok": True}
    assert client.get(f"/api/sessions/{sid}").json()["project_id"] == pid
    assert client.put(f"/api/sessions/{sid}/project", params={"pid": ""}).json() == {"ok": True}
    assert client.get(f"/api/sessions/{sid}").json()["project_id"] is None
    assert client.put(f"/api/sessions/{sid}/project").json() == {"ok": True}  # pid defaults to ""


def test_set_session_project_non_numeric_pid_errors(client):
    sid = client.post("/api/sessions", json={}).json()["id"]
    with pytest.raises(ValueError):  # original: int("abc") unhandled -> 500
        client.put(f"/api/sessions/{sid}/project", params={"pid": "abc"})


# ---------------------------------------------------------------- projects

def test_project_crud(client, tmp_path):
    assert client.get("/api/projects").json() == []
    pid = client.post("/api/projects", json={"name": "zeta", "path": str(tmp_path)}).json()["id"]
    client.post("/api/projects", json={"name": "alpha", "path": str(tmp_path)})
    rows = client.get("/api/projects").json()
    assert [p["name"] for p in rows] == ["alpha", "zeta"]  # ORDER BY name
    assert set(rows[0]) == {"id", "name", "path"}
    assert client.delete(f"/api/projects/{pid}").json() == {"ok": True}
    assert [p["name"] for p in client.get("/api/projects").json()] == ["alpha"]


def test_create_project_rejects_non_directory(client, tmp_path):
    bad = tmp_path / "missing"
    r = client.post("/api/projects", json={"name": "x", "path": str(bad)})
    assert r.status_code == 400
    assert r.json() == {"detail": f"not a directory: {bad}"}


def test_create_project_requires_fields(client):
    assert client.post("/api/projects", json={"name": "x"}).status_code == 422


@pytest.fixture
def proj(client, tmp_path):
    root = tmp_path / "proj"
    for d in ("node_modules", "sub", "build"):
        (root / d).mkdir(parents=True)
    (root / "a.py").write_text("def alpha(): return 'Banana'\n")
    (root / "sub" / "B.md").write_text("# beta\n")
    (root / "notes.txt").write_text("banana")  # extension not in CODE_EXT
    (root / "node_modules" / "x.js").write_text("banana")
    (root / "build" / "y.py").write_text("banana")
    pid = client.post("/api/projects", json={"name": "p", "path": str(root)}).json()["id"]
    return pid, root


def test_project_files_lists_code_files_sorted_skipping_junk(client, proj):
    pid, _ = proj
    assert client.get(f"/api/projects/{pid}/files").json() == ["a.py", "sub/B.md"]


def test_project_files_filter_is_case_insensitive_substring(client, proj):
    pid, _ = proj
    assert client.get(f"/api/projects/{pid}/files", params={"q": "SUB/b"}).json() == ["sub/B.md"]
    assert client.get(f"/api/projects/{pid}/files", params={"q": "zzz"}).json() == []


def test_project_files_unknown_project_404(client):
    r = client.get("/api/projects/999/files")
    assert r.status_code == 404 and r.json() == {"detail": "no such project"}


def test_project_grep_matches_content_case_insensitively(client, proj):
    pid, _ = proj
    assert client.get(f"/api/projects/{pid}/grep", params={"q": "banana"}).json() == ["a.py"]
    assert client.get(f"/api/projects/{pid}/grep", params={"q": "BETA"}).json() == ["sub/B.md"]


def test_project_grep_blank_query_short_circuits_before_lookup(client, proj):
    pid, _ = proj
    assert client.get(f"/api/projects/{pid}/grep", params={"q": "   "}).json() == []
    assert client.get("/api/projects/999/grep", params={"q": " "}).json() == []
    assert client.get("/api/projects/999/grep", params={"q": "x"}).status_code == 404


def test_project_grep_requires_q(client, proj):
    assert client.get(f"/api/projects/{proj[0]}/grep").status_code == 422


def test_scan_project_limit_and_filter(tmp_path):
    for n in "cab":
        (tmp_path / f"{n}.py").write_text("x")
    assert server.scan_project(tmp_path) == ["a.py", "b.py", "c.py"]
    assert len(server.scan_project(tmp_path, limit=2)) == 2
    assert server.scan_project(tmp_path, q="B.P") == ["b.py"]


@pytest.mark.xfail(strict=True, reason="behaviour change vs HEAD: original project_files skipped any file "
                   "whose ABSOLUTE path contained a SKIP_DIRS part (project under .../build/); "
                   "new scan_project checks only the project-relative path")
def test_project_files_in_root_under_skip_dir_name_matches_original(client, tmp_path):
    root = tmp_path / "build" / "app"
    root.mkdir(parents=True)
    (root / "a.py").write_text("x")
    pid = client.post("/api/projects", json={"name": "p", "path": str(root)}).json()["id"]
    assert client.get(f"/api/projects/{pid}/files").json() == []  # original result


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
