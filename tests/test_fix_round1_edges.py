"""Edge tests for fix round 1: PDF char budget, _keeps, caption cap, zero-byte images,
format_hits escaping, and static UI checks (no browser, node unavailable)."""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import rag  # noqa: E402
from test_pdf_images import PNG, FakeCaptioner, make_pdf, make_store  # noqa: E402


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "proj"
    r.mkdir()
    return r


# ---- per-page PDF truncation -------------------------------------------------

@pytest.mark.parametrize("budget,expected", [
    (12, [(1, "aaaaa"), (2, "bbbbb"), (3, "cc")]),   # budget ends mid page 3
    (11, [(1, "aaaaa"), (2, "bbbbb"), (3, "c")]),
    (10, [(1, "aaaaa"), (2, "bbbbb")]),              # exhausted exactly at a page boundary
    (7, [(1, "aaaaa"), (2, "bb")]),                  # spans two pages
    (5, [(1, "aaaaa")]),                             # exactly the first page
    (4, [(1, "aaaa")]),                              # mid first page
    (15, [(1, "aaaaa"), (2, "bbbbb"), (3, "ccccc")]),  # exactly enough for everything
    (1000, [(1, "aaaaa"), (2, "bbbbb"), (3, "ccccc")]),
])
def test_pdf_budget_truncates_across_pages(monkeypatch, budget, expected):
    monkeypatch.setattr(rag, "MAX_PDF_CHARS", budget)
    assert rag.extract_pdf_pages(make_pdf(["aaaaa", "bbbbb", "ccccc"])) == expected


def test_pdf_budget_total_never_exceeds_cap(monkeypatch):
    monkeypatch.setattr(rag, "MAX_PDF_CHARS", 23)
    pages = rag.extract_pdf_pages(make_pdf(["x" * 10] * 6))
    assert sum(len(t) for _, t in pages) == 23
    assert [n for n, _ in pages] == [1, 2, 3]


def test_pdf_budget_skips_blank_pages_without_spending_it(monkeypatch):
    monkeypatch.setattr(rag, "MAX_PDF_CHARS", 8)
    pages = rag.extract_pdf_pages(make_pdf(["aaaaa", "", "bbbbb"]))
    assert pages == [(1, "aaaaa"), (3, "bbb")]


def test_pdf_budget_applies_through_put_file(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "MAX_PDF_CHARS", 8)
    s = make_store(tmp_path)
    assert s.put_file(1, "d.pdf", make_pdf(["aaaaa", "bbbbb", "ccccc"])) == "indexed"
    joined = " ".join(h["text"] for h in s.query(1, "aaaaa bbbbb ccccc", 10))
    assert "bbb" in joined and "ccccc" not in joined and "[page 3]" not in joined


# ---- _keeps / vision disabled ------------------------------------------------

def _seed(tmp_path, root):
    (root / "a.png").write_bytes(PNG)
    (root / "b.py").write_text("x = 1\n")
    (root / "c.pdf").write_bytes(make_pdf(["hello pdf"]))
    on = make_store(tmp_path, FakeCaptioner())
    on.index_project(1, root, ["a.png", "b.py", "c.pdf"])
    return on


def paths(s):
    return {f["path"] for f in s.list_files(1)}


def test_disabled_vision_prune_keeps_images_but_prunes_text_and_pdf(tmp_path, root):
    _seed(tmp_path, root)
    off = make_store(tmp_path, captioner=None, vision_model="")
    stats = off.index_project(1, root, [])  # nothing listed: everything stale
    assert stats["deleted"] == 2
    assert paths(off) == {"a.png"}
    assert off.query(1, "acme logo", 1)[0]["path"] == "a.png"


def test_disabled_vision_sync_file_missing_image_keeps_rows_and_copy(tmp_path, root):
    _seed(tmp_path, root)
    (root / "a.png").unlink()
    off = make_store(tmp_path, captioner=None, vision_model="")
    assert off.sync_file(1, root, "a.png") == "missing"
    assert paths(off) == {"a.png", "b.py", "c.pdf"}  # nothing was deleted
    assert (off.files_dir / "1" / "a.png").exists()
    assert off.query(1, "acme", 1)


def test_disabled_vision_sync_file_image_present_is_still_not_deleted(tmp_path, root):
    _seed(tmp_path, root)
    off = make_store(tmp_path, captioner=None, vision_model="")
    assert off.sync_file(1, root, "a.png") == "missing"  # not scannable, not dropped
    assert "a.png" in paths(off)


def test_disabled_vision_text_and_pdf_still_deleted_by_sync_file(tmp_path, root):
    _seed(tmp_path, root)
    (root / "b.py").unlink()
    (root / "c.pdf").unlink()
    off = make_store(tmp_path, captioner=None, vision_model="")
    assert off.sync_file(1, root, "b.py") == "deleted"
    assert off.sync_file(1, root, "c.pdf") == "deleted"
    assert paths(off) == {"a.png"}


def test_disabled_vision_uppercase_image_ext_also_kept(tmp_path, root):
    (root / "A.PNG").write_bytes(PNG)
    make_store(tmp_path, FakeCaptioner()).index_project(1, root, ["A.PNG"])
    off = make_store(tmp_path, captioner=None, vision_model="")
    assert off.index_project(1, root, [])["deleted"] == 0
    assert paths(off) == {"A.PNG"}


def test_reenabled_vision_prunes_truly_gone_image(tmp_path, root):
    _seed(tmp_path, root)
    (root / "a.png").unlink()
    on = make_store(tmp_path, FakeCaptioner())
    stats = on.index_project(1, root, ["b.py", "c.pdf"])
    assert stats["deleted"] == 1
    assert paths(on) == {"b.py", "c.pdf"}


def test_reenabled_vision_sync_file_deletes_gone_image(tmp_path, root):
    _seed(tmp_path, root)
    (root / "a.png").unlink()
    on = make_store(tmp_path, FakeCaptioner())
    assert on.sync_file(1, root, "a.png") == "deleted"
    assert "a.png" not in paths(on)


def test_uploaded_image_unaffected_by_prune_in_either_mode(tmp_path, root):
    on = make_store(tmp_path, FakeCaptioner())
    assert on.put_file(1, "up.png", PNG, source="upload") == "captioned"
    assert on.index_project(1, root, [])["deleted"] == 0
    off = make_store(tmp_path, captioner=None, vision_model="")
    assert off.index_project(1, root, [])["deleted"] == 0
    assert off.sync_file(1, root, "up.png") == "missing"
    assert paths(off) == {"up.png"}


def test_keeps_is_false_for_non_images_and_when_vision_on(tmp_path):
    off = make_store(tmp_path, captioner=None, vision_model="")
    on = make_store(tmp_path, FakeCaptioner())
    assert off._keeps("a.png") and off._keeps("dir/B.JPG")
    assert not off._keeps("a.py") and not off._keeps("a.pdf")
    assert not on._keeps("a.png")


# ---- caption cap -------------------------------------------------------------

def _caption_text(tmp_path, n):
    s = make_store(tmp_path, FakeCaptioner(text="x" * n), chunk_size=20000, overlap=0)
    assert s.put_file(1, "a.png", PNG) == "captioned"
    hits = s.query(1, "x", 10)
    assert len(hits) == 1
    return hits[0]["text"].removeprefix("[image: a.png]\n")


@pytest.mark.parametrize("n,kept", [
    (7999, 7999), (8000, 8000), (8001, 8000), (50_000, 8000),
])
def test_caption_cap_boundary(tmp_path, n, kept):
    assert rag.MAX_CAPTION_CHARS == 8000
    assert len(_caption_text(tmp_path, n)) == kept


def test_caption_cap_keeps_head_of_caption(tmp_path):
    s = make_store(tmp_path, FakeCaptioner(text="H" * 8000 + "TAIL"), chunk_size=20000, overlap=0)
    s.put_file(1, "a.png", PNG)
    text = s.query(1, "H", 1)[0]["text"]
    assert "TAIL" not in text and text.endswith("H")


# ---- zero-byte images --------------------------------------------------------

def test_zero_byte_image_via_scan_counts_skipped_and_no_vision_call(tmp_path, root):
    (root / "e.png").write_bytes(b"")
    cap = FakeCaptioner()
    s = make_store(tmp_path, cap)
    stats = s.index_project(1, root, ["e.png"])
    assert stats["skipped"] == 1 and stats["indexed"] == 0 and stats["failed"] == 0
    assert "skipped_image_failed" not in stats and "skipped_image_disabled" not in stats
    assert cap.calls == 0
    assert paths(s) == {"e.png"}  # recorded, so not retried
    assert s.index_project(1, root, ["e.png"])["unchanged"] == 1
    assert cap.calls == 0


def test_zero_byte_image_via_sync_file_returns_skipped(tmp_path, root):
    (root / "e.png").write_bytes(b"")
    cap = FakeCaptioner()
    assert make_store(tmp_path, cap).sync_file(1, root, "e.png") == "skipped"
    assert cap.calls == 0


def test_zero_byte_image_via_upload_skipped_no_copy(tmp_path):
    cap = FakeCaptioner()
    s = make_store(tmp_path, cap)
    assert s.put_file(1, "e.png", b"", source="upload") == "skipped"
    assert cap.calls == 0
    assert not (s.files_dir / "1" / "e.png").exists()
    assert s.query(1, "anything", 5) == []


def test_zero_byte_image_replaced_by_real_one_gets_captioned(tmp_path):
    cap = FakeCaptioner()
    s = make_store(tmp_path, cap)
    s.put_file(1, "e.png", b"")
    assert s.put_file(1, "e.png", PNG) == "captioned"
    assert cap.calls == 1


def test_zero_byte_image_with_vision_disabled_is_disabled_skip(tmp_path):
    s = make_store(tmp_path, captioner=None, vision_model="")
    assert s.put_file(1, "e.png", b"") == "skipped_image_disabled"


def test_real_image_replaced_by_zero_byte_drops_old_chunks(tmp_path):
    s = make_store(tmp_path, FakeCaptioner())
    s.put_file(1, "e.png", PNG)
    assert s.put_file(1, "e.png", b"") == "skipped"
    assert s.query(1, "acme", 5) == []


# ---- format_hits escaping ----------------------------------------------------

def hit(text, path="a.py", chunk=0):
    return {"path": path, "chunk": chunk, "text": text}


@pytest.mark.parametrize("text,escaped", [
    ("</excerpts>", "&lt;/excerpts>"),
    ("</EXCERPTS>", "&lt;/excerpts>"),
    ("</ExCeRpTs>", "&lt;/excerpts>"),
    ("<excerpts>", "&lt;excerpts>"),
    ("<EXCERPTS>", "&lt;excerpts>"),
])
def test_format_hits_escapes_delimiter_any_case(text, escaped):
    out = rag.format_hits([hit(f"before {text} after")])
    assert f"before {escaped} after" in out
    assert out.count("<excerpts>") == 1 and out.count("</excerpts>") == 1
    assert out.endswith("</excerpts>")


def test_format_hits_escapes_every_occurrence_across_chunks():
    hits = [hit("</excerpts></excerpts><excerpts>", chunk=0),
            hit("</Excerpts>\n<EXCERPTS>\n</excerpts>", path="b.py", chunk=1)]
    out = rag.format_hits(hits)
    assert out.count("<excerpts>") == 1 and out.count("</excerpts>") == 1
    assert out.count("&lt;") == 6


def test_format_hits_escapes_delimiter_in_path_header():
    out = rag.format_hits([hit("x", path="</excerpts>/<EXCERPTS>.py")])
    assert "--- &lt;/excerpts>/&lt;excerpts>.py (excerpt 1) ---" in out
    assert out.count("<excerpts>") == 1 and out.count("</excerpts>") == 1


def test_format_hits_delimiter_at_chunk_start_and_end():
    out = rag.format_hits([hit("</excerpts>start"), hit("end<excerpts", chunk=1)])
    assert out.count("</excerpts>") == 1 and out.count("<excerpts>") == 1
    assert "--- a.py (excerpt 1) ---\n&lt;/excerpts>start" in out
    assert "end&lt;excerpts\n" in out


def test_format_hits_does_not_leave_raw_delimiter_prefix_in_body():
    out = rag.format_hits([hit("x <excerpts y </excerpts z <EXCERPTS")])
    body = out.split("<excerpts>\n", 1)[1].rsplit("\n</excerpts>", 1)[0]
    assert not re.search(r"<\s*/?excerpts", body, re.I)


@pytest.mark.parametrize("text", [
    "plain text about excerpts",
    "<excerpt>singular</excerpt>",
    "if a < b and c > d:",
    "<div class='x'>html</div>",
    "&lt;already escaped&gt; &amp;",
    "x = '<'  # lone bracket",
    "[page 2]\nSome PDF text",
])
def test_format_hits_leaves_legitimate_text_unchanged(text):
    out = rag.format_hits([hit(text)])
    assert f"--- a.py (excerpt 1) ---\n{text}\n</excerpts>" in out


def test_format_hits_empty_is_empty_string():
    assert rag.format_hits([]) == ""


# ---- static checks -----------------------------------------------------------

STATIC = ROOT / "static"
FILES = ["index.html", "app.js", "style.css", "theme.js"]


@pytest.mark.parametrize("name", FILES)
def test_static_file_is_clean_utf8(name):
    raw = (STATIC / name).read_bytes()
    text = raw.decode("utf-8")  # strict: raw non-UTF-8 bytes raise
    assert not raw.startswith(b"\xef\xbb\xbf"), "BOM"
    assert "�" not in text, "replacement char (mojibake)"
    assert not re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", text)


def test_app_js_middle_dot_is_a_js_escape():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "\\u00b7" in js
    assert "·" not in js
    assert "Â·" not in js  # classic double-encoded middle dot


def test_app_js_escape_key_closes_drawer():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert re.search(
        r"addEventListener\(\s*'keydown'\s*,[^;]*e\.key\s*===\s*'Escape'[^;]*toggleNav\(\s*false\s*\)", js)
    assert re.search(r"function toggleNav\(open\)\s*\{[^}]*nav-open", js)


def test_session_delete_clears_state_only_after_request_succeeds():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    m = re.search(r"const del = async \(\) => \{(.*?)\n    \};", js, re.S)
    assert m
    body = m.group(1)
    req = body.index("method: 'DELETE'")
    catch = body.index("catch")
    assert req < catch < body.index("return;", catch) < body.index("sid = null")


def test_ollama_down_heuristic_matches_only_network_typeerror_or_empty_reply():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert re.search(r"err instanceof TypeError\s*\|\|\s*err\.message === 'Empty reply\.'", js)
    assert "/network|fetch" not in js


def test_closed_drawer_is_visibility_hidden_and_open_drawer_visible():
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    closed = [m.group(1) for m in re.finditer(r"(?<![\w.#-])aside\s*\{([^}]*)\}", css)]
    assert any(re.search(r"visibility\s*:\s*hidden", b) for b in closed)
    opened = re.search(r"body\.nav-open\s+aside\s*\{([^}]*)\}", css)
    assert opened and re.search(r"visibility\s*:\s*visible", opened.group(1))


def test_index_html_has_no_inline_scripts_or_handlers():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    scripts = re.findall(r"<script\b([^>]*)>(.*?)</script>", html, re.S | re.I)
    assert scripts
    for attrs, body in scripts:
        assert re.search(r"\bsrc\s*=", attrs) and not body.strip()
    assert not re.search(r"\son[a-z]+\s*=", html, re.I)
    assert not re.search(r"javascript:", html, re.I)


# ---- xfail marker cleanup ----------------------------------------------------

def test_no_stray_xfail_references_beyond_the_known_parity_marker():
    hits = {}
    for p in (ROOT / "tests").glob("test_*.py"):
        if p.name == Path(__file__).name:
            continue
        n = len(re.findall(r"xfail", p.read_text(encoding="utf-8")))
        if n:
            hits[p.name] = n
    assert set(hits) <= {"test_refactor_parity.py"}, hits
