"""Fix round 2: Python 3.11 compatibility guard, format_hits combos, session ordering,
and static checks for toggleNav focus handling and the Ollama-down heuristic."""
import ast
import re
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import app as server  # noqa: E402
import rag  # noqa: E402
from backend import database  # noqa: E402
from backend.routes import files as files_routes  # noqa: E402

PY_FILES = sorted([ROOT / "rag.py", ROOT / "app.py", *(ROOT / "backend").rglob("*.py")])
JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")


def _ids(paths):
    return [str(p.relative_to(ROOT)) for p in paths]


# ---- Python 3.11 compatibility ----------------------------------------------

def test_py_file_list_is_nonempty():
    assert ROOT / "rag.py" in PY_FILES and len(PY_FILES) >= 8


@pytest.mark.parametrize("path", PY_FILES, ids=_ids(PY_FILES))
def test_parses_as_python_311(path):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 11))


def fstring_expr_violations(src):
    tree = ast.parse(src)
    offsets = [0]
    for ln in src.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(ln.encode("utf-8")))
    raw = src.encode("utf-8")
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FormattedValue):
            v = node.value
            seg = raw[offsets[v.lineno - 1] + v.col_offset: offsets[v.end_lineno - 1] + v.end_col_offset].decode("utf-8")
            if "\\" in seg:
                bad.append((v.lineno, seg))
            for sub in ast.walk(v):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and chr(10) in sub.value:
                    bad.append((sub.lineno, repr(sub.value)))
    return bad


@pytest.mark.parametrize("path", PY_FILES, ids=_ids(PY_FILES))
def test_no_backslash_in_fstring_expression_parts(path):
    assert not fstring_expr_violations(path.read_text(encoding="utf-8"))


def test_backslash_guard_detects_a_violation():
    src = "y = f'{\"" + chr(92) + "n\".join([])}'" + chr(10)
    assert fstring_expr_violations(src)


def test_backslash_guard_allows_backslash_in_literal_part():
    assert not fstring_expr_violations("y = f'a" + chr(92) + "n{1}'" + chr(10))


# ---- format_hits combos -------------------------------------------------------

def hit(text, path="a.py", chunk=0):
    return {"path": path, "text": text, "chunk": chunk}


def test_format_hits_empty_is_blank():
    assert rag.format_hits([]) == ""


@pytest.mark.parametrize("path,text", [
    ("</excerpts>.py", "</excerpts>"),
    ("<EXCERPTS/x", "<Excerpts>"),
    ("</ExCeRpTs>", "plain"),
    ("plain.py", "<excerpts></excerpts><excerpts>"),
])
def test_format_hits_neutralises_path_and_text_combos(path, text):
    out = rag.format_hits([hit(text, path=path), hit(text, path=path, chunk=1)])
    assert out.count("<excerpts>") == 1 and out.count("</excerpts>") == 1
    assert out.startswith("Below are excerpts") and out.endswith("</excerpts>")


def test_format_hits_path_with_newline_and_special_chars_kept():
    out = rag.format_hits([hit("t", path="dir/a b\c.py", chunk=4)])
    assert "--- dir/a b\c.py (excerpt 5) ---\nt" in out


def test_format_hits_blocks_separated_by_blank_line_in_order():
    out = rag.format_hits([hit("one", chunk=0), hit("two", path="b.py", chunk=2)])
    assert "--- a.py (excerpt 1) ---\none\n\n--- b.py (excerpt 3) ---\ntwo\n</excerpts>" in out


# ---- session ordering ------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "server.db")
    up = tmp_path / "uploads"
    up.mkdir()
    monkeypatch.setattr(files_routes, "UPLOADS", up)
    server.init_db()
    return TestClient(server.app)


def test_rapid_session_creation_is_newest_first(client):
    ids = [client.post("/api/sessions", json={"title": f"s{i}"}).json()["id"] for i in range(25)]
    assert [r["id"] for r in client.get("/api/sessions").json()] == ids[::-1]


def test_identical_created_at_direct_insert_orders_by_id_desc(client):
    with database.db() as c:
        for i in range(6):
            c.execute("INSERT INTO sessions (title, project_id, created_at) VALUES (?,?,?)",
                      (f"t{i}", None, 5.0))
    rows = client.get("/api/sessions").json()
    ids = [r["id"] for r in rows]
    assert ids == sorted(ids, reverse=True) and len(ids) == 6


def test_created_at_dominates_id_in_ordering(client):
    with database.db() as c:
        c.execute("INSERT INTO sessions (title, project_id, created_at) VALUES ('old-high-id',NULL,1)")
        c.execute("INSERT INTO sessions (title, project_id, created_at) VALUES ('new-low-id',NULL,9)")
        c.execute("UPDATE sessions SET id = 1000 WHERE title = 'old-high-id'")
    assert [r["title"] for r in client.get("/api/sessions").json()] == ["new-low-id", "old-high-id"]


def test_mixed_ties_and_distinct_times(client):
    with database.db() as c:
        for t, ts in [("a", 2), ("b", 2), ("c", 3), ("d", 1), ("e", 2)]:
            c.execute("INSERT INTO sessions (title, project_id, created_at) VALUES (?,NULL,?)", (t, ts))
    assert [r["title"] for r in client.get("/api/sessions").json()] == ["c", "e", "b", "a", "d"]


# ---- static: toggleNav focus + Ollama-down heuristic -------------------------------

def _toggle_body():
    m = re.search(r"function toggleNav\(open\)\s*\{(.*?)\n\}", JS, re.S)
    assert m, "toggleNav not found"
    return m.group(1)


def test_toggle_nav_focuses_side_on_open():
    b = _toggle_body()
    assert re.search(r"if\s*\(open\)\s*\$\('side'\)\.querySelector\([^)]*button[^)]*\)\?\.focus\(\)", b)


def test_toggle_nav_restores_focus_to_menu_button_only_when_focus_inside_side():
    b = _toggle_body()
    assert re.search(r"else if\s*\(\$\('side'\)\.contains\(document\.activeElement\)\)\s*\$\('menuBtn'\)\.focus\(\)", b)


def test_toggle_nav_toggles_class_with_explicit_state():
    assert "document.body.classList.toggle('nav-open', open)" in _toggle_body()


def test_failure_message_only_replaced_when_no_text_arrived():
    # changed with the clean-errors work: a reply with text keeps it (partial); only an empty one is a failure
    assert re.search(r"if \(full\.trim\(\)\) \{.*?\} else \{.*?failReply\(", JS, re.S)


def test_side_and_menu_button_exist_in_html():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert re.search(r'id="side"', html) and re.search(r'id="menuBtn"', html)
