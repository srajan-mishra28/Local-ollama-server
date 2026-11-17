"""Fix-round-1 probes: upload name flattening, name length, OSError mapping, mid-stream delete, confirm()."""
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend import database, rag_store  # noqa: E402
from backend.routes import project_files  # noqa: E402
from backend.routes.project_files import upload_name  # noqa: E402
from test_projects import _FakeResp, client, new_project, ollama, upload  # noqa: E402,F401

ROOT = Path(__file__).resolve().parent.parent
B = chr(92)


@pytest.mark.parametrize("raw,expected", [
    ("a/b" + B + "c.md", "c.md"),
    ("a" + B + "b/c.md", "c.md"),
    (B + B + "server" + B + "share" + B + "x.md", "x.md"),
    ("C:" + B + "dir" + B + "x.md", "x.md"),
    ("dir/", "upload.txt"),
    ("dir" + B, "upload.txt"),
    ("/", "upload.txt"),
    ("", "upload.txt"),
    (None, "upload.txt"),
    ("a/..", ".."),
    ("a" + B + ".", "."),
    ("..", ".."),
])
def test_upload_name_flattening(raw, expected):
    assert upload_name(raw) == expected


@pytest.mark.parametrize("filename", ["a/b" + B + "c.md", B + B + "server" + B + "share" + B + "x.md"])
def test_upload_mixed_separators_end_to_end(client, filename):
    pid = new_project(client)
    res = upload(client, pid, **{filename: b"hello world"})
    assert res[0]["result"] == "indexed" and "/" not in res[0]["path"] and "\\" not in res[0]["path"]


def test_upload_trailing_separator_becomes_default_name(client):
    pid = new_project(client)
    res = upload(client, pid, **{"folder/": b"hello world"})
    assert res[0]["path"] == "upload.txt" and res[0]["result"] == "indexed"


@pytest.mark.parametrize("filename", ["a/..", "a" + B + ".", ".."])
def test_upload_dot_names_never_500_and_never_escape(client, tmp_path, filename):
    pid = new_project(client)
    before = {p for p in tmp_path.rglob("*")}
    res = upload(client, pid, **{filename: b"hello world", "ok.md": b"fine content"})
    assert res[0]["result"] == "error" and res[0]["status"] == 400
    assert res[1]["result"] == "indexed"
    new = {p for p in tmp_path.rglob("*")} - before
    assert all(tmp_path in p.parents for p in new)


@pytest.mark.parametrize("n,ok", [(200, True), (201, False)])
def test_name_length_boundary(client, monkeypatch, n, ok):
    monkeypatch.setattr(project_files, "MAX_NAME_LENGTH", 20)
    name = "x" * (n - 20 + 20 - 3 if False else 0) or None
    name = ("x" * (20 if ok else 21 - 3)) + ".md" if False else None
    length = 20 if ok else 21
    name = "x" * (length - 3) + ".md"
    assert len(name) == length
    pid = new_project(client)
    res = upload(client, pid, **{name: b"hello world", "ok.md": b"fine content"})
    if ok:
        assert res[0]["result"] == "indexed"
    else:
        assert res[0]["result"] == "error" and res[0]["status"] == 400
        assert "name too long" in res[0]["detail"]
    assert res[1]["result"] == "indexed"


def test_real_limit_is_200():
    assert project_files.MAX_NAME_LENGTH == 200


def test_name_length_counts_flattened_name_not_full_path(client, monkeypatch):
    monkeypatch.setattr(project_files, "MAX_NAME_LENGTH", 10)
    pid = new_project(client)
    res = upload(client, pid, **{"d" * 50 + "/a.md": b"hello world"})
    assert res[0]["result"] == "indexed" and res[0]["path"] == "a.md"


def test_upload_oserror_on_write_is_per_file_400(client, monkeypatch):
    real = Path.write_bytes

    def boom(self, data):
        if self.name == "bad.md":
            raise OSError("disk says no")
        return real(self, data)

    monkeypatch.setattr(Path, "write_bytes", boom)
    pid = new_project(client)
    res = upload(client, pid, **{"bad.md": b"hello world", "ok.md": b"fine content"})
    assert res[0]["result"] == "error" and res[0]["status"] == 400
    assert "not usable" in res[0]["detail"]
    assert res[1]["result"] == "indexed"
    assert [f["path"] for f in client.get(f"/api/projects/{pid}/files").json()] == ["ok.md"]


def test_delete_oserror_is_400(client, monkeypatch):
    pid = new_project(client)
    upload(client, pid, **{"a.md": b"hello world"})

    def boom(self, *a, **k):
        raise OSError("locked")

    monkeypatch.setattr(Path, "unlink", boom)
    r = client.delete(f"/api/projects/{pid}/files", params={"path": "a.md"})
    assert r.status_code == 400


def _msgs():
    with database.db() as c:
        return c.execute("SELECT role, content FROM messages ORDER BY id").fetchall()


def test_normal_stream_persists_exactly_one_full_assistant_message(client, ollama, monkeypatch):
    sid = client.post("/api/sessions", json={"title": "t"}).json()["id"]

    async def lines(self):
        for p in ["Hel", "lo ", "world"]:
            yield json.dumps({"message": {"content": p}})

    monkeypatch.setattr(_FakeResp, "aiter_lines", lines)
    r = client.post("/api/chat", json={"session_id": sid, "content": "hi"})
    assert r.text == "Hello world"
    rows = [tuple(x) for x in _msgs()]
    assert rows == [("user", "hi"), ("assistant", "Hello world")]


def test_deleting_sibling_session_mid_stream_does_not_affect_stream(client, ollama, monkeypatch):
    pid = new_project(client)
    s1 = client.post(f"/api/projects/{pid}/sessions", json={"title": "one"}).json()["id"]
    s2 = client.post(f"/api/projects/{pid}/sessions", json={"title": "two"}).json()["id"]

    async def lines(self):
        yield json.dumps({"message": {"content": "answer"}})
        with database.db() as c:
            c.execute("DELETE FROM messages WHERE session_id = ?", (s2,))
            c.execute("DELETE FROM sessions WHERE id = ?", (s2,))

    monkeypatch.setattr(_FakeResp, "aiter_lines", lines)
    r = client.post("/api/chat", json={"session_id": s1, "content": "hi"})
    assert r.text == "answer"
    with database.db() as c:
        rows = c.execute("SELECT session_id, role, content FROM messages").fetchall()
    assert [tuple(x) for x in rows] == [(s1, "user", "hi"), (s1, "assistant", "answer")]


def test_session_deleted_mid_stream_leaves_no_rows_for_it(client, ollama, monkeypatch):
    sid = client.post("/api/sessions", json={"title": "t"}).json()["id"]

    async def lines(self):
        yield json.dumps({"message": {"content": "a"}})
        yield json.dumps({"message": {"content": "b"}})
        with database.db() as c:
            c.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
            c.execute("DELETE FROM sessions WHERE id = ?", (sid,))

    monkeypatch.setattr(_FakeResp, "aiter_lines", lines)
    client.post("/api/chat", json={"session_id": sid, "content": "hi"})
    with database.db() as c:
        assert c.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (sid,)).fetchone()[0] == 0


def test_every_delete_fetch_is_preceded_by_confirm():
    js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    calls = list(re.finditer(r"method:\s*'DELETE'", js))
    assert len(calls) >= 4
    for mt in calls:
        # nearest preceding confirm( must be within the same function, before any other function boundary
        window = js[max(0, mt.start() - 400):mt.start()]
        assert "confirm(" in window, js[max(0, mt.start() - 80):mt.start() + 20]
        assert re.search(r"if\s*\(!confirm\([^\n]*\)\)\s*return", window)
