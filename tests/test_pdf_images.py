import io
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app as server  # noqa: E402
import rag  # noqa: E402
from backend import database, rag_store  # noqa: E402
from test_rag import fake_embedder  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def make_pdf(pages: list[str]) -> bytes:
    """A minimal hand-built PDF with one line of Helvetica text per page ('' = blank page)."""
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b""]
    kids = []
    font_id = 3 + 2 * len(pages)
    for i, text in enumerate(pages):
        page_id, content_id = 3 + 2 * i, 4 + 2 * i
        kids.append(f"{page_id} 0 R")
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] /Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()
        )
        stream = f"BT /F1 12 Tf 20 150 Td ({text}) Tj ET".encode() if text else b""
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(pages)} >>".encode()
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for n, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % n + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


def encrypted_pdf() -> bytes:
    w = PdfWriter()
    w.add_blank_page(100, 100)
    w.encrypt("secret", algorithm="RC4-128")
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


class FakeCaptioner:
    def __init__(self, text="a red logo reading ACME", fail=None):
        self.text, self.fail, self.calls = text, fail, 0

    def __call__(self, image: bytes) -> str:
        self.calls += 1
        if self.fail:
            raise self.fail
        return self.text


def make_store(tmp_path, captioner=None, vision_model="v1", **kw):
    return rag.RagStore(tmp_path / "rag.db", tmp_path / "copies", fake_embedder, model="fake",
                        captioner=captioner, vision_model=vision_model, **kw)


@pytest.fixture
def store(tmp_path):
    return make_store(tmp_path, FakeCaptioner())


# ---- PDFs ------------------------------------------------------------------

def test_pdf_text_with_page_markers(store):
    pdf = make_pdf(["alpha banana", "", "gamma quantum"])
    assert store.put_file(1, "doc.pdf", pdf) == "indexed"
    texts = [h["text"] for h in store.query(1, "quantum", 10)]
    assert any(t.startswith("[page 3]") and "quantum" in t for t in texts)
    assert any(t.startswith("[page 1]") for t in texts)
    assert not any("[page 2]" in t for t in texts)  # blank page has no chunk
    assert (store.files_dir / "1" / "doc.pdf").read_bytes() == pdf
    assert "[page 3]" in rag.format_hits(store.query(1, "quantum", 1))


def test_pdf_page_cap(tmp_path):
    s = make_store(tmp_path, max_pdf_pages=2)
    s.put_file(1, "d.pdf", make_pdf(["one", "two", "three"]))
    texts = " ".join(h["text"] for h in s.query(1, "one two three", 10))
    assert "[page 2]" in texts and "[page 3]" not in texts


def test_pdf_without_text_layer_skipped(store):
    assert store.put_file(1, "scan.pdf", make_pdf(["", ""])) == "skipped_pdf_no_text"
    assert not store.has_index(1)
    assert not (store.files_dir / "1" / "scan.pdf").exists()


@pytest.mark.parametrize("data", [b"not a pdf at all", b"%PDF-1.4\ngarbage", make_pdf(["x"])[:60]])
def test_corrupt_pdf_skipped(store, data):
    assert store.put_file(1, "bad.pdf", data) == "skipped_pdf_unreadable"


def test_encrypted_pdf_skipped(store):
    assert store.put_file(1, "enc.pdf", encrypted_pdf()) == "skipped_pdf_unreadable"


def test_pdf_size_cap(tmp_path):
    s = make_store(tmp_path, max_pdf_bytes=100)
    assert s.put_file(1, "big.pdf", make_pdf(["hello"])) == "skipped"
    assert s.max_bytes("a.pdf") == 100 and s.max_bytes("a.py") == rag.MAX_FILE_BYTES
    assert s.max_bytes("a.png") == rag.MAX_IMAGE_BYTES


# ---- images ----------------------------------------------------------------

def test_image_caption_indexed_with_header(store):
    assert store.put_file(1, "img/logo.png", PNG) == "captioned"
    hit = store.query(1, "acme logo", 1)[0]
    assert hit["text"].startswith("[image: img/logo.png]") and "ACME" in hit["text"]
    assert (store.files_dir / "1" / "img" / "logo.png").read_bytes() == PNG


def test_captioner_failure_skips_without_recording(tmp_path):
    cap = FakeCaptioner(fail=rag.CaptionError("model missing"))
    s = make_store(tmp_path, cap)
    assert s.put_file(1, "a.png", PNG) == "skipped_image_failed"
    assert s.list_files(1) == []  # not recorded, so a later run retries
    cap.fail = None
    assert s.put_file(1, "a.png", PNG) == "captioned"


def test_vision_disabled(tmp_path):
    s = make_store(tmp_path, captioner=None, vision_model="")
    assert s.put_file(1, "a.png", PNG) == "skipped_image_disabled"
    assert s.list_files(1) == []


def test_image_signature_includes_vision_model(tmp_path):
    cap = FakeCaptioner()
    make_store(tmp_path, cap, "v1").put_file(1, "a.png", PNG)
    assert make_store(tmp_path, cap, "v1").put_file(1, "a.png", PNG) == "unchanged"
    assert cap.calls == 1
    assert make_store(tmp_path, cap, "v2").put_file(1, "a.png", PNG) == "captioned"
    assert cap.calls == 2


def test_image_size_cap(tmp_path):
    cap = FakeCaptioner()
    s = make_store(tmp_path, cap, max_image_bytes=10)
    assert s.put_file(1, "a.png", PNG) == "skipped"
    assert cap.calls == 0


def test_ollama_captioner_errors(monkeypatch):
    cap = rag.OllamaCaptioner("http://x", "vm")

    def reply(status, body):
        return lambda *a, **k: httpx.Response(status, json=body, request=httpx.Request("POST", "http://x"))

    monkeypatch.setattr(httpx, "post", reply(200, {"message": {"content": " a cat "}}))
    assert cap(PNG) == "a cat"
    monkeypatch.setattr(httpx, "post", reply(404, {"error": "model not found"}))
    with pytest.raises(rag.CaptionError, match="ollama pull vm"):
        cap(PNG)
    monkeypatch.setattr(httpx, "post", reply(200, {"nope": 1}))
    with pytest.raises(rag.CaptionError):
        cap(PNG)
    monkeypatch.setattr(httpx, "post", reply(200, {"message": {"content": "  "}}))
    with pytest.raises(rag.CaptionError):
        cap(PNG)

    def refuse(*a, **k):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "post", refuse)
    with pytest.raises(httpx.ConnectError):
        cap(PNG)


# ---- API -------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "server.db")
    server.init_db()
    monkeypatch.setattr(rag_store, "store", make_store(tmp_path, FakeCaptioner()))
    return TestClient(server.app)


@pytest.fixture
def pid(client):
    return client.post("/api/projects", json={"name": "p"}).json()["id"]


def upload(client, pid, name, data):
    """Upload one file; returns its entry of the per-file results."""
    r = client.post(f"/api/projects/{pid}/files", files={"files": (name, data)})
    assert r.status_code == 200
    return r.json()["results"][0]


def test_upload_indexes_text_pdf_and_png(client, pid):
    assert upload(client, pid, "notes.md", b"plain notes") == {"path": "notes.md", "result": "indexed"}
    assert upload(client, pid, "manual.pdf", make_pdf(["install the flux capacitor"]))["result"] == "indexed"
    assert upload(client, pid, "shot.png", PNG) == {"path": "shot.png", "result": "captioned"}
    files = client.get(f"/api/projects/{pid}/files").json()
    assert {f["path"] for f in files} == {"notes.md", "manual.pdf", "shot.png"}
    assert all(f["chunks"] >= 1 for f in files)
    hits = client.get(f"/api/projects/{pid}/rag", params={"q": "flux capacitor"}).json()
    assert hits[0]["path"] == "manual.pdf" and hits[0]["text"].startswith("[page 1]")


def test_upload_pdf_without_text_is_skipped(client, pid):
    assert upload(client, pid, "s.pdf", make_pdf([""]))["result"] == "skipped_pdf_no_text"


def test_upload_caps_are_per_type(client, pid, monkeypatch):
    monkeypatch.setattr(rag_store.store, "max_pdf_bytes", 50)
    assert upload(client, pid, "a.pdf", make_pdf(["hello"]))["status"] == 413
    big = b"a " * (rag.MAX_FILE_BYTES // 2 + 10)  # over the text cap
    assert upload(client, pid, "a.txt", big)["status"] == 413
    # a PDF may exceed the 1 MB text cap
    monkeypatch.setattr(rag_store.store, "max_pdf_bytes", rag.MAX_PDF_BYTES)
    pdf = make_pdf(["hello"]) + b"\n%" + b"x" * (rag.MAX_FILE_BYTES + 10)
    assert upload(client, pid, "big.pdf", pdf)["result"] == "indexed"


def test_vision_disabled_keeps_existing_captions(tmp_path):
    on = make_store(tmp_path, FakeCaptioner())
    on.put_file(1, "a.png", PNG)
    off = make_store(tmp_path, captioner=None, vision_model="")
    assert {f["path"] for f in off.list_files(1)} == {"a.png"}
    assert "ACME" in off.query(1, "acme logo", 1)[0]["text"]


def test_zero_byte_image_skipped_without_vision_call(tmp_path):
    cap = FakeCaptioner()
    s = make_store(tmp_path, cap)
    assert s.put_file(1, "e.png", b"") == "skipped"
    assert s.put_file(1, "e.png", b"") == "unchanged"
    assert cap.calls == 0


def test_caption_length_is_capped(tmp_path):
    s = make_store(tmp_path, FakeCaptioner(text="word " * 10_000))
    s.put_file(1, "a.png", PNG)
    chars = sum(len(h["text"]) for h in s.query(1, "word", 100))
    assert chars < rag.MAX_CAPTION_CHARS + 1000  # headers and overlap only


def test_pdf_char_cap_truncates_page(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "MAX_PDF_CHARS", 10)
    pages = rag.extract_pdf_pages(make_pdf(["abcdefghijklmnopqrstuvwxyz", "second"]))
    assert pages == [(1, "abcdefghij")]
