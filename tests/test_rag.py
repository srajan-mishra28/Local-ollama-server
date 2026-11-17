import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rag  # noqa: E402


def fake_embedder(texts):
    """Deterministic bag-of-words style embedding: hash words into 64 buckets."""
    out = []
    for t in texts:
        v = np.zeros(64)
        for w in t.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        out.append(v.tolist())
    return out


@pytest.fixture
def store(tmp_path):
    return rag.RagStore(tmp_path / "rag.db", tmp_path / "copies", fake_embedder,
                        model="fake", chunk_size=200, overlap=40)


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "proj"
    r.mkdir()
    (r / "a.py").write_text("def alpha():\n    return 'banana smoothie'\n")
    (r / "b.py").write_text("def beta():\n    return 'quantum entanglement'\n")
    return r


def index(store, pid, root, rels):
    """Put files from a directory into the store (the fixtures write real files)."""
    return [store.put_file(pid, r, (root / r).read_bytes()) for r in rels]


def test_chunk_overlap_and_size():
    text = "".join(f"line {i:03d}\n" for i in range(100))
    chunks = rag.chunk_text(text, size=100, overlap=30)
    assert len(chunks) > 5
    assert all(len(c) <= 100 for c in chunks)
    # next chunk begins with the tail of the previous one
    assert chunks[1].splitlines()[0] in chunks[0].splitlines()
    assert "".join(chunks).count("line 099") >= 1


def test_chunk_long_line_and_empty():
    assert rag.chunk_text("") == []
    assert rag.chunk_text("   \n\n") == []
    assert all(len(c) <= 50 for c in rag.chunk_text("x" * 500, size=50, overlap=10))
    with pytest.raises(ValueError):
        rag.chunk_text("a", size=10, overlap=10)


def test_decode_text():
    assert rag.decode_text(b"hello") == "hello"
    assert rag.decode_text(b"\x00\x01binary") is None
    assert rag.decode_text(b"\xff\xfe\xfa") is None
    assert rag.decode_text(b"") is None


@pytest.mark.parametrize("bad", ["", "../x", "a/../../x", "/etc/passwd"])
def test_clean_relpath_rejects(bad):
    with pytest.raises(ValueError):
        rag.clean_relpath(bad)


def test_clean_relpath_normalises():
    assert rag.clean_relpath(r"src\a.py") == "src/a.py"


def test_index_and_query(store, root):
    assert index(store, 1, root, ["a.py", "b.py"]) == ["indexed", "indexed"]
    hits = store.query(1, "banana smoothie", k=2)
    assert hits[0]["path"] == "a.py"
    assert hits[0]["score"] > hits[1]["score"]
    assert (store.files_dir / "1" / "a.py").read_text() == (root / "a.py").read_text()


def test_query_scoped_to_project(store, root):
    index(store, 1, root, ["a.py"])
    assert store.query(2, "banana") == []
    assert not store.has_index(2)


def test_reupload_unchanged_and_changed(store, root):
    assert index(store, 1, root, ["a.py"]) == ["indexed"]
    assert index(store, 1, root, ["a.py"]) == ["unchanged"]
    (root / "a.py").write_text("def alpha():\n    return 'cherry pie'\n")
    assert index(store, 1, root, ["a.py"]) == ["indexed"]
    assert store.query(1, "cherry pie")[0]["text"].count("cherry") == 1
    assert all("banana" not in h["text"] for h in store.query(1, "banana"))  # no stale chunks


def test_delete_file(store, root):
    index(store, 1, root, ["a.py", "b.py"])
    assert store.delete_file(1, "a.py") is True
    assert store.delete_file(1, "a.py") is False
    assert all(h["path"] == "b.py" for h in store.query(1, "banana", k=10))


def test_binary_skipped_gracefully(store):
    assert store.put_file(1, "img.bin", b"\x00\x01\x02") == "skipped"
    assert not store.has_index(1)


def test_binary_replacing_text_drops_chunks(store):
    store.put_file(1, "x.txt", b"some text here")
    assert store.put_file(1, "x.txt", b"\x00binary") == "skipped"
    assert not store.has_index(1)


def test_embedder_failure_keeps_old_chunks(tmp_path, root):
    calls = {"fail": False}

    def flaky(texts):
        if calls["fail"]:
            raise RuntimeError("ollama down")
        return fake_embedder(texts)

    s = rag.RagStore(tmp_path / "r.db", tmp_path / "c", flaky, model="m")
    s.put_file(1, "a.py", b"original content")
    calls["fail"] = True
    with pytest.raises(RuntimeError):
        s.put_file(1, "a.py", b"new content")
    calls["fail"] = False
    assert s.query(1, "original")[0]["text"] == "original content"


def test_model_change_triggers_reindex(tmp_path):
    a = rag.RagStore(tmp_path / "r.db", tmp_path / "c", fake_embedder, model="m1")
    a.put_file(1, "a.py", b"same content")
    b = rag.RagStore(tmp_path / "r.db", tmp_path / "c", fake_embedder, model="m2")
    assert b.put_file(1, "a.py", b"same content") == "indexed"


def test_delete_project(store, root):
    index(store, 1, root, ["a.py"])
    store.delete_project(1)
    assert not store.has_index(1) and not (store.files_dir / "1").exists()


def test_query_blank_or_k_zero(store, root):
    index(store, 1, root, ["a.py"])
    assert store.query(1, "  ") == []
    assert store.query(1, "banana", k=0) == []


def test_format_hits():
    assert rag.format_hits([]) == ""
    out = rag.format_hits([{"path": "a.py", "chunk": 0, "text": "code", "score": 1}])
    assert "a.py" in out and "code" in out
