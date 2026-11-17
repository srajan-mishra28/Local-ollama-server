"""Edge cases for PDF / image indexing, size caps, path safety and the API around them."""
import io
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader, PdfWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app as server  # noqa: E402
import rag  # noqa: E402
from backend import database, rag_store  # noqa: E402
from test_pdf_images import PNG, FakeCaptioner, make_pdf, make_store  # noqa: E402

def texts(store, pid=1, q="x", k=50):
    return [h["text"] for h in store.query(pid, q, k)]


def empty_password_pdf() -> bytes:
    w = PdfWriter(clone_from=PdfReader(io.BytesIO(make_pdf(["open sesame"]))))
    w.encrypt("", algorithm="RC4-128")
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


# ---- PDF edge cases ----------------------------------------------------------

def test_every_pdf_chunk_starts_with_its_own_page_marker(tmp_path):
    s = make_store(tmp_path, chunk_size=60, overlap=5)
    long1 = " ".join(f"alpha{i}" for i in range(30))
    long2 = " ".join(f"beta{i}" for i in range(30))
    assert s.put_file(1, "d.pdf", make_pdf([long1, long2])) == "indexed"
    chunks = texts(s)
    assert len(chunks) > 2  # each page was split into several chunks
    for c in chunks:
        assert c.startswith(("[page 1]\n", "[page 2]\n"))
        assert c.count("[page ") == 1  # a chunk never spans two pages
        body = c.split("\n", 1)[1]
        assert ("alpha" in body) != ("beta" in body)
    assert any(c.startswith("[page 1]") for c in chunks) and any(c.startswith("[page 2]") for c in chunks)


def test_pdf_chunk_indexes_are_sequential_across_pages(tmp_path):
    s = make_store(tmp_path)
    s.put_file(1, "d.pdf", make_pdf(["one", "two", "three"]))
    hits = s.query(1, "one two three", 10)
    assert sorted(h["chunk"] for h in hits) == [0, 1, 2]


def test_pdf_page_limit_exact_boundary(tmp_path):
    s = make_store(tmp_path, max_pdf_pages=3)
    s.put_file(1, "d.pdf", make_pdf(["p1", "p2", "p3", "p4", "p5"]))
    joined = " ".join(texts(s))
    assert "[page 3]" in joined and "[page 4]" not in joined and "[page 5]" not in joined


def test_pdf_page_limit_counts_blank_pages(tmp_path):
    # the limit is on page numbers, not on pages with text
    s = make_store(tmp_path, max_pdf_pages=2)
    assert s.put_file(1, "d.pdf", make_pdf(["", "", "late text"])) == "skipped_pdf_no_text"


def test_extract_pdf_pages_char_guard(monkeypatch):
    monkeypatch.setattr(rag, "MAX_PDF_CHARS", 10)
    pages = rag.extract_pdf_pages(make_pdf(["aaaaaaaa"] * 5))  # 8 chars per page
    assert [n for n, _ in pages] == [1, 2]  # stops once over the guard


def test_extract_pdf_pages_returns_numbers_of_pages_with_text():
    pages = rag.extract_pdf_pages(make_pdf(["first", "", "third"]))
    assert [n for n, _ in pages] == [1, 3]


def test_empty_password_encrypted_pdf_is_indexed(tmp_path):
    s = make_store(tmp_path)
    assert s.put_file(1, "e.pdf", empty_password_pdf()) == "indexed"
    assert any("sesame" in t for t in texts(s))


def test_truncated_pdf_does_not_raise(tmp_path):
    s = make_store(tmp_path)
    pdf = make_pdf(["hello world"])
    result = s.put_file(1, "t.pdf", pdf[: len(pdf) // 2])
    assert result in {"skipped_pdf_unreadable", "skipped_pdf_no_text", "indexed"}
    assert result != "indexed" or s.has_index(1)


@pytest.mark.parametrize("data", [b"", b"%PDF-", b"\x00" * 100])
def test_degenerate_pdf_bytes_are_skipped(tmp_path, data):
    s = make_store(tmp_path)
    assert s.put_file(1, "z.pdf", data).startswith("skipped")
    assert not s.has_index(1)


def test_whitespace_only_pdf_is_no_text(tmp_path):
    s = make_store(tmp_path)
    assert s.put_file(1, "w.pdf", make_pdf(["   ", "  "])) == "skipped_pdf_no_text"
    assert not (s.files_dir / "1" / "w.pdf").exists()


def test_pdf_becoming_textless_drops_old_chunks_and_copy(tmp_path):
    s = make_store(tmp_path)
    s.put_file(1, "d.pdf", make_pdf(["keep me"]))
    assert s.has_index(1)
    assert s.put_file(1, "d.pdf", make_pdf([""])) == "skipped_pdf_no_text"
    assert not s.has_index(1)
    assert not (s.files_dir / "1" / "d.pdf").exists()


def test_pdf_uppercase_extension_is_pdf(tmp_path):
    assert rag.file_kind("A/B.PDF") == "pdf"
    s = make_store(tmp_path)
    assert s.put_file(1, "B.PDF", make_pdf(["upper"])) == "indexed"


def test_unchanged_pdf_not_reparsed(tmp_path):
    s = make_store(tmp_path)
    pdf = make_pdf(["same"])
    assert s.put_file(1, "d.pdf", pdf) == "indexed"
    assert s.put_file(1, "d.pdf", pdf) == "unchanged"


# ---- image edge cases ---------------------------------------------------------

@pytest.mark.parametrize("name", ["a.png", "a.jpg", "a.jpeg", "a.webp", "a.gif", "a.bmp",
                                  "A.PNG", "b.JPG", "c.JpEg"])
def test_each_image_extension_is_captioned(tmp_path, name):
    cap = FakeCaptioner("a blue square")
    s = make_store(tmp_path, cap)
    assert rag.file_kind(name) == "image"
    assert s.put_file(1, name, PNG) == "captioned"
    assert cap.calls == 1
    assert texts(s)[0].startswith(f"[image: {name}]")


def test_non_image_extension_never_reaches_captioner(tmp_path):
    cap = FakeCaptioner()
    s = make_store(tmp_path, cap)
    s.put_file(1, "a.tiff", PNG)  # not in IMAGE_EXT: treated as (binary) text
    assert cap.calls == 0


def test_empty_caption_yields_no_chunks(tmp_path):
    s = make_store(tmp_path, FakeCaptioner(""))
    assert s.put_file(1, "a.png", PNG) == "skipped"
    assert not s.has_index(1)
    assert not (s.files_dir / "1" / "a.png").exists()


def test_whitespace_caption_yields_no_chunks(tmp_path):
    s = make_store(tmp_path, FakeCaptioner("  \n "))
    assert s.put_file(1, "a.png", PNG) == "skipped"
    assert not s.has_index(1)


def test_long_caption_is_chunked_with_header_on_each(tmp_path):
    s = make_store(tmp_path, FakeCaptioner("\n".join(f"line {i} " + "w" * 20 for i in range(30))),
                   chunk_size=100, overlap=10)
    s.put_file(1, "a.png", PNG)
    chunks = texts(s)
    assert len(chunks) > 1 and all(c.startswith("[image: a.png]\n") for c in chunks)


@pytest.mark.parametrize("exc", [ValueError("x"), KeyError("k"), RuntimeError("r")])
def test_unexpected_caption_exception_propagates_from_put_file(tmp_path, exc):
    s = make_store(tmp_path, FakeCaptioner(fail=exc))
    with pytest.raises(type(exc)):
        s.put_file(1, "a.png", PNG)
    assert s.list_files(1) == []


def test_ollama_captioner_exception_mapping(monkeypatch):
    cap = rag.OllamaCaptioner("http://x", "vm")

    def raiser(exc):
        def f(*a, **k):
            raise exc
        return f

    monkeypatch.setattr(httpx, "post", raiser(httpx.ReadTimeout("slow")))
    with pytest.raises(rag.CaptionError, match="timed out"):
        cap(PNG)
    for exc in (httpx.ConnectTimeout("t"), httpx.ConnectError("c")):
        monkeypatch.setattr(httpx, "post", raiser(exc))
        with pytest.raises(httpx.TransportError):
            cap(PNG)

    def status(code):
        return lambda *a, **k: httpx.Response(code, json={}, request=httpx.Request("POST", "http://x"))

    monkeypatch.setattr(httpx, "post", status(500))
    with pytest.raises(rag.CaptionError, match="500") as ei:
        cap(PNG)
    assert "ollama pull" not in str(ei.value)
    monkeypatch.setattr(httpx, "post", lambda *a, **k: httpx.Response(
        200, content=b"not json", request=httpx.Request("POST", "http://x")))
    with pytest.raises(rag.CaptionError):
        cap(PNG)
    monkeypatch.setattr(httpx, "post", lambda *a, **k: httpx.Response(
        200, json={"message": {"content": None}}, request=httpx.Request("POST", "http://x")))
    with pytest.raises(rag.CaptionError):
        cap(PNG)


def test_ollama_captioner_sends_base64_image_and_model(monkeypatch):
    seen = {}

    def post(url, json, timeout):
        seen.update(url=url, json=json, timeout=timeout)
        return httpx.Response(200, json={"message": {"content": "ok"}}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    assert rag.OllamaCaptioner("http://h:1", "vm", 7)(PNG) == "ok"
    assert seen["url"] == "http://h:1/api/chat" and seen["timeout"] == 7
    assert seen["json"]["model"] == "vm" and seen["json"]["stream"] is False
    import base64
    assert base64.b64decode(seen["json"]["messages"][0]["images"][0]) == PNG


def test_model_change_replaces_caption_chunks(tmp_path):
    s1 = make_store(tmp_path, FakeCaptioner("old caption zebra"), "v1")
    s1.put_file(1, "a.png", PNG)
    s2 = make_store(tmp_path, FakeCaptioner("new caption walrus"), "v2")
    assert s2.put_file(1, "a.png", PNG) == "captioned"
    joined = " ".join(texts(s2))
    assert "walrus" in joined and "zebra" not in joined


def test_vision_model_change_does_not_recaption_text_or_pdf(tmp_path):
    cap = FakeCaptioner()
    make_store(tmp_path, cap, "v1").put_file(1, "a.py", b"x = 1\n")
    make_store(tmp_path, cap, "v1").put_file(1, "a.pdf", make_pdf(["hi"]))
    s2 = make_store(tmp_path, cap, "v2")
    assert s2.put_file(1, "a.py", b"x = 1\n") == "unchanged"
    assert s2.put_file(1, "a.pdf", make_pdf(["hi"])) == "unchanged"


def test_vision_disabled_image_upload_is_skipped_not_error(tmp_path):
    s = make_store(tmp_path, captioner=None, vision_model="")
    assert s.put_file(1, "a.png", PNG) == "skipped_image_disabled"
    assert s.list_files(1) == [] and not (s.files_dir / "1" / "a.png").exists()


def test_store_without_captioner_but_vision_model_set_still_disabled(tmp_path):
    s = make_store(tmp_path, captioner=None, vision_model="v1")
    assert s.put_file(1, "a.jpg", PNG) == "skipped_image_disabled"


# ---- size caps (store level) -----------------------------------------------

def test_cap_boundary_is_inclusive(tmp_path):
    pdf = make_pdf(["edge"])
    assert make_store(tmp_path, max_pdf_bytes=len(pdf)).put_file(1, "e.pdf", pdf) == "indexed"
    assert make_store(tmp_path / "b", max_pdf_bytes=len(pdf) - 1).put_file(1, "e.pdf", pdf) == "skipped"


def test_oversize_image_put_file_is_skipped_without_captioning(tmp_path):
    s = make_store(tmp_path, max_image_bytes=4)
    assert s.put_file(1, "a.png", PNG) == "skipped"


# ---- path safety -------------------------------------------------

BAD_NAMES = ["C:/x.pdf", "C:\\x.png", "a.pdf:stream", "a.png:Zone.Identifier", "CON.png", "con.pdf",
             "sub/NUL.png", "COM1.pdf", "../x.pdf", "a/../../x.png", "/abs/x.pdf", "x.pdf.", "x.png ",
             "", "   "]


@pytest.mark.parametrize("bad", BAD_NAMES)
def test_put_file_rejects_unsafe_binary_names(tmp_path, bad):
    s = make_store(tmp_path, FakeCaptioner())
    data = make_pdf(["x"]) if "pdf" in bad else PNG
    with pytest.raises(ValueError):
        s.put_file(1, bad, data)
    assert s.list_files(1) == []
    assert not any(p.is_file() for p in (tmp_path / "copies").rglob("*"))


def test_symlink_copy_cannot_escape(tmp_path):
    s = make_store(tmp_path, FakeCaptioner())
    outside = tmp_path / "outside"
    outside.mkdir()
    base = s.files_dir / "1"
    base.mkdir(parents=True)
    try:
        (base / "link").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted")
    with pytest.raises(ValueError):
        s.put_file(1, "link/x.png", PNG)
    assert not any(outside.iterdir())


# ---- delete / cleanup ------------------------------------------------------

def test_delete_file_removes_pdf_and_image_copies(tmp_path):
    s = make_store(tmp_path, FakeCaptioner())
    s.put_file(1, "docs/a.pdf", make_pdf(["x"]))
    s.put_file(1, "img/a.png", PNG)
    for rel in ("docs/a.pdf", "img/a.png"):
        assert (s.files_dir / "1" / rel).exists()
        assert s.delete_file(1, rel) is True
        assert not (s.files_dir / "1" / rel).exists()
        assert s.delete_file(1, rel) is False
    assert not s.has_index(1) and s.list_files(1) == []


def test_delete_project_removes_binary_copies_only_for_that_project(tmp_path):
    s = make_store(tmp_path, FakeCaptioner())
    s.put_file(1, "a.pdf", make_pdf(["x"]))
    s.put_file(1, "a.png", PNG)
    s.put_file(2, "a.png", PNG)
    s.delete_project(1)
    assert not (s.files_dir / "1").exists()
    assert (s.files_dir / "2" / "a.png").exists() and s.has_index(2)


# ---- stats back-compat --------------------------------------------------------

# ---- scan_project / build_context_block ---------------------------------------

# ---- API ------------------------------------------------------------------------


def test_failed_caption_keeps_old_index(tmp_path):
    cap = FakeCaptioner("original caption")
    s = make_store(tmp_path, cap)
    s.put_file(1, "a.png", PNG)
    # the image changes and the model now fails: the previous caption must survive
    cap.fail = rag.CaptionError("busy")
    assert s.put_file(1, "a.png", PNG + b"extra") == "skipped_image_failed"
    assert [f["path"] for f in s.list_files(1)] == ["a.png"]
    assert any("original caption" in t for t in texts(s))
    assert (s.files_dir / "1" / "a.png").read_bytes() == PNG


def test_failed_caption_is_retried_on_next_upload(tmp_path):
    cap = FakeCaptioner(fail=rag.CaptionError("no"))
    s = make_store(tmp_path, cap)
    assert s.put_file(1, "a.png", PNG) == "skipped_image_failed"
    assert s.put_file(1, "a.png", PNG) == "skipped_image_failed"
    assert cap.calls == 2  # not remembered as "unchanged"
    cap.fail = None
    assert s.put_file(1, "a.png", PNG) == "captioned"
    assert s.put_file(1, "a.png", PNG) == "unchanged"


@pytest.mark.parametrize("bad", ["C:/x.pdf", "a.pdf:stream", "CON.png", "../x.pdf"])
def test_delete_file_rejects_unsafe_names(tmp_path, bad):
    s = make_store(tmp_path, FakeCaptioner())
    with pytest.raises(ValueError):
        s.delete_file(1, bad)


# ---- upload API --------------------------------------------------------------

def _status_error(code=500):
    req = httpx.Request("POST", "http://x")
    return httpx.HTTPStatusError("boom", request=req, response=httpx.Response(code, request=req))


def _client(tmp_path, monkeypatch, captioner, vision="v1", **kw):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "server.db")
    server.init_db()
    monkeypatch.setattr(rag_store, "store", make_store(tmp_path, captioner, vision, **kw))
    return TestClient(server.app)


def _project(c):
    return c.post("/api/projects", json={"name": "p"}).json()["id"]


def _up(c, pid, name, data):
    """Upload one file; returns its entry of the per-file results."""
    r = c.post(f"/api/projects/{pid}/files", files={"files": (name, data)})
    assert r.status_code == 200
    return r.json()["results"][0]


def _no_copies(tmp_path):
    return not any(p.is_file() for p in (tmp_path / "copies").rglob("*"))


def test_upload_vision_disabled_is_a_skipped_result(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch, None, "")
    assert _up(c, _project(c), "a.png", PNG) == {"path": "a.png", "result": "skipped_image_disabled"}


@pytest.mark.parametrize("fail", [httpx.ConnectError("down"), _status_error()])
def test_upload_ollama_failure_is_a_502_entry(tmp_path, monkeypatch, fail):
    c = _client(tmp_path, monkeypatch, FakeCaptioner(fail=fail))
    res = _up(c, _project(c), "a.png", PNG)
    assert res["result"] == "error" and res["status"] == 502


def test_upload_caption_error_is_skipped(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch, FakeCaptioner(fail=rag.CaptionError("no")))
    assert _up(c, _project(c), "a.png", PNG)["result"] == "skipped_image_failed"


@pytest.mark.parametrize("name,kw,data", [
    ("a.pdf", {"max_pdf_bytes": 40}, make_pdf(["hello"])),
    ("a.PNG", {"max_image_bytes": 8}, PNG),
    ("a.jpg", {"max_image_bytes": 8}, PNG),
    ("a.py", {}, b"#" * (rag.MAX_FILE_BYTES + 1)),
])
def test_upload_over_cap_is_413_entry_and_leaves_no_state(tmp_path, monkeypatch, name, kw, data):
    c = _client(tmp_path, monkeypatch, FakeCaptioner(), **kw)
    pid = _project(c)
    res = _up(c, pid, name, data)
    assert res["result"] == "error" and res["status"] == 413
    assert c.get(f"/api/projects/{pid}/files").json() == []
    assert _no_copies(tmp_path)


def test_upload_exactly_at_cap_accepted_one_over_rejected(tmp_path, monkeypatch):
    pdf = make_pdf(["edge"])
    c = _client(tmp_path, monkeypatch, FakeCaptioner(), max_pdf_bytes=len(pdf))
    pid = _project(c)
    assert _up(c, pid, "a.pdf", pdf)["result"] == "indexed"
    assert _up(c, pid, "b.pdf", pdf + b"%")["status"] == 413


def test_upload_image_above_text_cap_but_under_image_cap_ok(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch, FakeCaptioner())
    big = PNG + b"\x00" * (rag.MAX_FILE_BYTES + 10)
    assert _up(c, _project(c), "big.png", big)["result"] == "captioned"


@pytest.mark.parametrize("filename", ["a.pdf:stream", "CON.png", "COM3.pdf", "sub/NUL.pdf", "x.png."])
def test_upload_unsafe_filename_is_a_400_entry(tmp_path, monkeypatch, filename):
    c = _client(tmp_path, monkeypatch, FakeCaptioner())
    pid = _project(c)
    res = _up(c, pid, filename, PNG)
    assert res["result"] == "error" and res["status"] == 400
    assert c.get(f"/api/projects/{pid}/files").json() == []
    assert _no_copies(tmp_path)


@pytest.mark.parametrize("filename,expected", [
    ("C:/x.pdf", "x.pdf"), ("C:\\dir\\x.pdf", "x.pdf"), ("../x.pdf", "x.pdf"), ("/abs/x.pdf", "x.pdf"),
])
def test_upload_filename_is_reduced_to_basename(tmp_path, monkeypatch, filename, expected):
    c = _client(tmp_path, monkeypatch, FakeCaptioner())
    res = _up(c, _project(c), filename, make_pdf(["hi"]))
    assert res == {"path": expected, "result": "indexed"}


def test_delete_file_and_delete_project_clean_copies(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch, FakeCaptioner())
    pid = _project(c)
    _up(c, pid, "a.pdf", make_pdf(["hi"]))
    _up(c, pid, "a.png", PNG)
    copies = tmp_path / "copies" / str(pid)
    assert (copies / "a.pdf").exists() and (copies / "a.png").exists()
    assert c.delete(f"/api/projects/{pid}/files", params={"path": "a.png"}).json() == {"deleted": True}
    assert not (copies / "a.png").exists() and (copies / "a.pdf").exists()
    assert c.delete(f"/api/projects/{pid}").status_code == 200
    assert not copies.exists()
