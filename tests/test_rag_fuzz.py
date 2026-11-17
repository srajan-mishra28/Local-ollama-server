import random
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rag  # noqa: E402
from test_rag import fake_embedder  # noqa: E402

ALPHABET = ["a", "b", " ", "\n", "\r\n", "\r", "\t", "é", "日", "😀", "x" * 7]


def _random_text(rng):
    return "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 400)))


@pytest.mark.parametrize("seed", range(40))
def test_chunk_text_fuzz_size_bound_and_coverage(seed):
    rng = random.Random(seed)
    size = rng.randint(2, 60)
    overlap = rng.randint(0, size - 1)
    text = _random_text(rng)

    chunks = rag.chunk_text(text, size, overlap)

    assert all(len(c) <= size for c in chunks)
    assert all(c.strip() for c in chunks)
    # no non-whitespace character is lost
    joined = "".join(chunks)
    assert all(ch in joined for ch in set(text) if not ch.isspace())


@pytest.mark.parametrize("seed", range(20))
def test_chunk_text_fuzz_first_and_last_content_present(seed):
    rng = random.Random(1000 + seed)
    size = rng.randint(5, 50)
    text = _random_text(rng)
    chunks = rag.chunk_text(text, size, rng.randint(0, size - 1))
    if not text.strip():
        assert chunks == []
        return
    assert chunks[0].lstrip().startswith(text.lstrip()[:1])
    assert chunks[-1].rstrip().endswith(text.rstrip()[-1:])


@pytest.mark.parametrize("size,overlap", [(1, 0), (2, 1), (10, 9)])
def test_chunk_text_tiny_sizes_respect_bound(size, overlap):
    text = "ab\ncd\n" * 20 + "z" * 37
    assert all(len(c) <= size for c in rag.chunk_text(text, size, overlap))


@pytest.mark.parametrize("size,overlap", [(5, 5), (5, 6)])
def test_chunk_text_rejects_overlap_ge_size(size, overlap):
    with pytest.raises(ValueError):
        rag.chunk_text("hello", size, overlap)


@pytest.mark.parametrize("ok", ["./a.py", "a//b.py", "a/./b.py", "CONSOLE/x.py", " lead/x.py"])
def test_clean_relpath_normalises_or_accepts_odd_but_safe(ok):
    out = rag.clean_relpath(ok)
    assert not out.startswith("/") and ".." not in out.split("/")


@pytest.mark.parametrize("bad", ["CON.tar.gz", "a/Aux.b.c", "...", "a/.../b", "x/ ", r"\\?\C:\x", r"a\..\b"])
def test_clean_relpath_rejects_more_windows_forms(bad):
    with pytest.raises(ValueError):
        rag.clean_relpath(bad)


@pytest.fixture
def store(tmp_path):
    return rag.RagStore(tmp_path / "r.db", tmp_path / "copies", fake_embedder, model="m")


def test_shadowed_scan_does_not_embed_or_touch_copy(tmp_path):
    calls = []

    def emb(texts):
        calls.append(texts)
        return fake_embedder(texts)

    s = rag.RagStore(tmp_path / "r.db", tmp_path / "copies", emb, model="m")
    assert s.put_file(1, "n.md", b"uploaded", "upload") == "indexed"
    n = len(calls)
    assert s.put_file(1, "n.md", b"scanned", "scan") == "shadowed"
    assert len(calls) == n
    assert (tmp_path / "copies" / "1" / "n.md").read_bytes() == b"uploaded"


def test_scan_prune_never_removes_upload(store, tmp_path):
    store.put_file(1, "u.md", b"mine", "upload")
    stats = store.index_project(1, tmp_path, [], prune=True)
    assert stats["deleted"] == 0
    assert store.has_index(1)


def test_sync_file_missing_upload_is_kept(store, tmp_path):
    store.put_file(1, "u.md", b"mine", "upload")
    assert store.sync_file(1, tmp_path, "u.md") == "missing"
    assert store.list_files(1)[0]["source"] == "upload"


def test_upload_can_be_deleted_explicitly(store):
    store.put_file(1, "u.md", b"mine", "upload")
    assert store.delete_file(1, "u.md") is True
    assert not (store.files_dir / "1" / "u.md").exists()


def test_abort_does_not_prune_even_with_prune_true(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    (root / "b.py").write_text("y = 2\n")
    ok = rag.RagStore(tmp_path / "r.db", tmp_path / "c", fake_embedder, model="m")
    ok.index_project(1, root, ["a.py", "b.py"])

    def down(texts):
        raise httpx.ConnectError("down")

    s = rag.RagStore(tmp_path / "r.db", tmp_path / "c", down, model="m2")
    stats = s.index_project(1, root, ["a.py", "b.py"], prune=True)
    assert "aborted" in stats and stats["failed"] == 1
    assert {f["path"] for f in s.list_files(1)} == {"a.py", "b.py"}


def test_http_status_error_does_not_abort(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    (root / "b.py").write_text("y = 2\n")

    def bad(texts):
        raise httpx.HTTPStatusError("500", request=httpx.Request("POST", "http://x"),
                                    response=httpx.Response(500))

    s = rag.RagStore(tmp_path / "r.db", tmp_path / "c", bad, model="m")
    stats = s.index_project(1, root, ["a.py", "b.py"])
    assert stats["failed"] == 2 and "aborted" not in stats


def test_query_blank_does_not_raise_reindex_on_stale_index(tmp_path):
    a = rag.RagStore(tmp_path / "r.db", tmp_path / "c", fake_embedder, model="old")
    a.put_file(1, "a.md", b"hello world")
    b = rag.RagStore(tmp_path / "r.db", tmp_path / "c", fake_embedder, model="new")
    assert b.query(1, "  ") == []


def test_query_other_project_stale_index_is_independent(tmp_path):
    a = rag.RagStore(tmp_path / "r.db", tmp_path / "c", fake_embedder, model="old")
    a.put_file(1, "a.md", b"hello world")
    b = rag.RagStore(tmp_path / "r.db", tmp_path / "c", fake_embedder, model="new")
    assert b.query(2, "hello") == []  # project 2 has nothing: not a reindex case
    with pytest.raises(rag.ReindexRequired):
        b.query(1, "hello")


def test_reindex_after_model_change_clears_reindex_required(tmp_path):
    a = rag.RagStore(tmp_path / "r.db", tmp_path / "c", fake_embedder, model="old")
    a.put_file(1, "a.md", b"hello world")
    b = rag.RagStore(tmp_path / "r.db", tmp_path / "c", fake_embedder, model="new")
    assert b.put_file(1, "a.md", b"hello world") == "indexed"
    assert b.query(1, "hello")[0]["path"] == "a.md"
    assert len(b.list_files(1)) == 1


def test_format_hits_empty_is_blank():
    assert rag.format_hits([]) == ""
