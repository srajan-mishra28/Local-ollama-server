"""Fix round 1 follow-up: Reply.finish, first_piece polling, piece parsing, markers, retrieve,
title_from_message, failReply and repo-wide merge-conflict checks. No network, no Ollama."""
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import app as server  # noqa: E402
from backend import database  # noqa: E402
from backend.database import db  # noqa: E402
from backend.routes import chat as chat_mod  # noqa: E402
from backend.routes import files as files_routes  # noqa: E402
from backend.routes import sessions as sess  # noqa: E402


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "server.db")
    up = tmp_path / "uploads"
    up.mkdir()
    monkeypatch.setattr(files_routes, "UPLOADS", up)
    server.init_db()
    monkeypatch.setattr(chat_mod, "DISCONNECT_POLL_SECONDS", 0.01)
    with db() as c:
        sid = c.execute("INSERT INTO sessions (title, created_at) VALUES ('New chat', 1)").lastrowid
    return sid


def rows(sid):
    with db() as c:
        return [(r["role"], r["content"]) for r in c.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY id", (sid,))]


class FakePieces:
    """Async-generator stand-in; counts aclose calls. script items: str | Exception | float (sleep)."""
    def __init__(self, script):
        self.script = list(script)
        self.closed = 0
        self.gen = self._run()

    async def _run(self):
        for it in self.script:
            if isinstance(it, float):
                await asyncio.sleep(it)
            elif isinstance(it, BaseException):
                raise it
            else:
                yield it

    def __aiter__(self):
        return self

    def __anext__(self):
        return self.gen.__anext__()

    async def aclose(self):
        self.closed += 1
        await self.gen.aclose()


class Req:
    def __init__(self, gone_after=None):
        self.calls = 0
        self.gone_after = gone_after

    async def is_disconnected(self):
        self.calls += 1
        return self.gone_after is not None and self.calls > self.gone_after


def patch_pieces(monkeypatch, fake):
    monkeypatch.setattr(chat_mod, "ollama_pieces", lambda messages: fake)


async def drain(resp):
    out = []
    async for p in resp.body_iterator:
        out.append(p)
    return out


def run_chat(sid, request=None, text="hello"):
    return chat_mod.chat(chat_mod.ChatIn(session_id=sid, content=text), request)


def add_user(sid, content="q", ts=1):
    with db() as c:
        return c.execute("INSERT INTO messages (session_id,role,content,created_at) VALUES (?,?,?,?)",
                         (sid, "user", content, ts)).lastrowid


# ---- Reply.finish -------------------------------------------------------------

def test_finish_concurrent_calls_persist_once(env):
    sid = env
    mid = add_user(sid)

    async def go():
        r = chat_mod.Reply(sid, mid)
        fake = FakePieces(["a"])
        r.pieces = fake
        r.collected = ["par", "tial"]
        await asyncio.gather(*(r.finish() for _ in range(5)))
        return fake

    fake = asyncio.run(go())
    assert fake.closed == 1
    assert rows(sid) == [("user", "q"), ("assistant", "partial")]


def test_finish_empty_drops_only_that_user_message_once(env):
    sid = env
    add_user(sid, "old", 1)
    b = add_user(sid, "new", 2)

    async def go():
        r = chat_mod.Reply(sid, b)
        await r.finish()
        await r.finish()
    asyncio.run(go())
    assert rows(sid) == [("user", "old")]


def test_finish_after_session_deleted_saves_nothing_and_does_not_raise(env):
    sid = env

    async def go():
        r = chat_mod.Reply(sid, 999)
        r.collected = ["x"]
        with db() as c:
            c.execute("DELETE FROM sessions WHERE id=?", (sid,))
        await r.finish()
    asyncio.run(go())
    assert rows(sid) == []


def test_finish_save_raises_propagates_and_is_not_retried(env, monkeypatch):
    sid = env
    calls = []

    def boom(*a):
        calls.append(a)
        raise RuntimeError("disk")
    monkeypatch.setattr(chat_mod, "save_reply", boom)

    async def go():
        r = chat_mod.Reply(sid, 1)
        r.collected = ["x"]
        with pytest.raises(RuntimeError):
            await r.finish()
        await r.finish()  # second call is a no-op
    asyncio.run(go())
    assert len(calls) == 1


def test_finish_cancelled_during_aclose_still_closes_upstream(env):
    sid = env

    class Slow(FakePieces):
        async def aclose(self):
            await asyncio.sleep(0.05)
            await super().aclose()

    async def go():
        r = chat_mod.Reply(sid, 1)
        fake = Slow(["a"])
        r.pieces = fake
        r.collected = ["t"]
        import anyio
        with anyio.CancelScope() as scope:  # how Starlette/anyio cancels a request
            async def later():
                await asyncio.sleep(0.01)
                scope.cancel()
            asyncio.ensure_future(later())
            await r.finish()
        return fake
    fake = asyncio.run(go())
    assert fake.closed == 1  # shielded: cancel did not interrupt teardown


def test_finish_when_aclose_raises_marks_done(env):
    sid = env

    class Bad(FakePieces):
        async def aclose(self):
            self.closed += 1
            raise RuntimeError("x")

    async def go():
        r = chat_mod.Reply(sid, 1)
        r.pieces = Bad([])
        with pytest.raises(RuntimeError):
            await r.finish()
        await r.finish()
        return r.pieces
    assert asyncio.run(go()).closed == 1


# ---- chat paths: exactly-once persistence, upstream closed once ---------------------

def test_stream_normal_stores_once_closes_once(env, monkeypatch):
    fake = FakePieces(["a", "b", "c"])
    patch_pieces(monkeypatch, fake)

    async def go():
        resp = await run_chat(env, Req())
        out = await drain(resp)
        await resp.background()  # background task after the body
        return out
    assert asyncio.run(go()) == ["a", "b", "c"]
    assert rows(env) == [("user", "hello"), ("assistant", "abc")]
    assert fake.closed == 1


def test_response_discarded_unstarted_background_finishes_once(env, monkeypatch):
    fake = FakePieces(["first", "more"])
    patch_pieces(monkeypatch, fake)

    async def go():
        resp = await run_chat(env, Req())
        await resp.background()
        await resp.background()
    asyncio.run(go())
    assert rows(env) == [("user", "hello"), ("assistant", "first")]
    assert fake.closed == 1


def test_stop_mid_stream_keeps_partial_once(env, monkeypatch):
    fake = FakePieces(["a", "b", "c"])
    patch_pieces(monkeypatch, fake)

    async def go():
        resp = await run_chat(env, Req())
        it = resp.body_iterator
        await it.__anext__()
        await it.__anext__()
        await it.aclose()  # client Stop
        await resp.background()
    asyncio.run(go())
    assert rows(env) == [("user", "hello"), ("assistant", "ab")]
    assert fake.closed == 1


def test_error_before_first_token_is_502_and_user_message_dropped(env, monkeypatch):
    fake = FakePieces([chat_mod.OllamaStreamError("model exploded")])
    patch_pieces(monkeypatch, fake)
    with pytest.raises(chat_mod.HTTPException) as ei:
        asyncio.run(run_chat(env, Req()))
    assert ei.value.status_code == 502 and "model exploded" in ei.value.detail
    assert rows(env) == []
    assert fake.closed == 1


def test_connect_error_before_first_token_is_502(env, monkeypatch):
    fake = FakePieces([httpx.ConnectError("no")])
    patch_pieces(monkeypatch, fake)
    with pytest.raises(chat_mod.HTTPException) as ei:
        asyncio.run(run_chat(env, None))
    assert ei.value.status_code == 502 and "Can't reach Ollama" in ei.value.detail
    assert rows(env) == [] and fake.closed == 1


def test_empty_upstream_is_502_and_dropped(env, monkeypatch):
    fake = FakePieces([])
    patch_pieces(monkeypatch, fake)
    with pytest.raises(chat_mod.HTTPException) as ei:
        asyncio.run(run_chat(env, Req()))
    assert ei.value.status_code == 502
    assert rows(env) == [] and fake.closed == 1


def test_unexpected_exception_before_first_token_cleans_up_and_propagates(env, monkeypatch):
    fake = FakePieces([ValueError("weird")])
    patch_pieces(monkeypatch, fake)
    with pytest.raises(ValueError):
        asyncio.run(run_chat(env, Req()))
    assert rows(env) == [] and fake.closed == 1


def test_error_mid_stream_marker_streamed_and_stored_once(env, monkeypatch):
    fake = FakePieces(["partial", chat_mod.OllamaStreamError("oom")])
    patch_pieces(monkeypatch, fake)

    async def go():
        resp = await run_chat(env, Req())
        out = await drain(resp)
        await resp.background()
        return out
    out = asyncio.run(go())
    assert out[0] == "partial"
    assert out[1] == "\n\n[Reply interrupted: Ollama returned an error: oom]"
    assert rows(env) == [("user", "hello"), ("assistant", "".join(out))]
    assert fake.closed == 1


def test_transport_error_mid_stream_marker(env, monkeypatch):
    fake = FakePieces(["x", httpx.ReadError("cut")])
    patch_pieces(monkeypatch, fake)

    async def go():
        resp = await run_chat(env, Req())
        return await drain(resp)
    out = asyncio.run(go())
    assert out[-1].startswith("\n\n[Reply interrupted: Lost the connection")
    assert rows(env)[-1][1].endswith("Is it still running?]")


def test_clean_early_close_has_no_marker(env, monkeypatch):
    fake = FakePieces(["only"])
    patch_pieces(monkeypatch, fake)

    async def go():
        resp = await run_chat(env, Req())
        return await drain(resp)
    assert asyncio.run(go()) == ["only"]
    assert rows(env)[-1] == ("assistant", "only")


def test_disconnect_before_first_token_returns_204_and_drops(env, monkeypatch):
    fake = FakePieces([5.0, "late"])
    patch_pieces(monkeypatch, fake)
    resp = asyncio.run(run_chat(env, Req(gone_after=2)))
    assert resp.status_code == 204
    assert rows(env) == [] and fake.closed == 1


def test_disconnect_with_instant_piece_leaves_consistent_store(env, monkeypatch):
    """The first piece and the disconnect race; whichever wins must leave a consistent store."""
    fake = FakePieces(["now"])
    patch_pieces(monkeypatch, fake)

    async def go():
        resp = await run_chat(env, Req(gone_after=0))
        if resp.status_code != 204:
            await resp.background()
        return resp
    resp = asyncio.run(go())
    if resp.status_code == 204:
        assert rows(env) == []
    else:
        assert rows(env) == [("user", "hello"), ("assistant", "now")]
    assert fake.closed == 1


def test_slow_first_token_over_many_polls_still_streams(env, monkeypatch):
    fake = FakePieces([0.15, "ok"])
    patch_pieces(monkeypatch, fake)
    req = Req()

    async def go():
        resp = await run_chat(env, req)
        return await drain(resp)
    assert asyncio.run(go()) == ["ok"]
    assert req.calls >= 5
    assert rows(env)[-1] == ("assistant", "ok")


def test_request_none_path_streams(env, monkeypatch):
    fake = FakePieces(["z"])
    patch_pieces(monkeypatch, fake)

    async def go():
        return await drain(await run_chat(env, None))
    assert asyncio.run(go()) == ["z"]


def test_first_piece_leaks_no_tasks(env):
    async def go():
        base = set(asyncio.all_tasks())
        for gone, script in ((None, ["a"]), (1, [5.0]), (None, [chat_mod.OllamaStreamError("e")])):
            fake = FakePieces(script)
            try:
                await chat_mod.first_piece(fake, Req(gone_after=gone))
            except (chat_mod.ClientGone, chat_mod.OllamaStreamError):
                pass
            await fake.aclose()
        await asyncio.sleep(0)
        return set(asyncio.all_tasks()) - base
    assert asyncio.run(go()) == set()


def test_full_chat_leaves_no_tasks(env, monkeypatch):
    fake = FakePieces(["a", "b"])
    patch_pieces(monkeypatch, fake)

    async def go():
        base = set(asyncio.all_tasks())
        resp = await run_chat(env, Req())
        await drain(resp)
        await resp.background()
        await asyncio.sleep(0)
        return set(asyncio.all_tasks()) - base
    assert asyncio.run(go()) == set()


def test_cancellation_during_first_piece_cleans_up(env, monkeypatch):
    fake = FakePieces([5.0, "x"])
    patch_pieces(monkeypatch, fake)

    async def go():
        t = asyncio.ensure_future(run_chat(env, Req()))
        await asyncio.sleep(0.05)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
    asyncio.run(go())
    assert rows(env) == [] and fake.closed == 1


def test_http_end_to_end_via_testclient(env, monkeypatch):
    from fastapi.testclient import TestClient
    fake = FakePieces(["he", "llo"])
    patch_pieces(monkeypatch, fake)
    r = TestClient(server.app).post("/api/chat", json={"session_id": env, "content": "hi"})
    assert r.status_code == 200 and r.text == "hello"
    assert rows(env) == [("user", "hi"), ("assistant", "hello")]
    assert fake.closed == 1


# ---- piece_from_line ------------------------------------------------------------

@pytest.mark.parametrize("line", [
    "null", "1", "12345678901234567890123456789012345678901234567890", "true", '"str"', "[]",
    '[{"message":{"content":"x"}}]', "{}", '{"message":null}', '{"message":[]}', '{"message":"x"}',
    '{"message":{}}', '{"message":{"content":null}}', '{"message":{"content":5}}',
    '{"message":{"content":["a"]}}', '{"error":null}', '{"error":5}', '{"error":{"m":1}}',
    '{"error":["x"]}', "﻿", "﻿{not json", "{", "   ", "NaN", "-Infinity", "[" * 1000,
    '﻿{"message":{"content":"x"}}',
])
def test_piece_from_line_ignores_odd_shapes(line):
    assert chat_mod.piece_from_line(line) == ""


def test_piece_from_line_valid_and_error():
    assert chat_mod.piece_from_line('{"message":{"content":"hi"}}') == "hi"
    assert chat_mod.piece_from_line('{"message":{"content":""}}') == ""
    with pytest.raises(chat_mod.OllamaStreamError) as ei:
        chat_mod.piece_from_line(json.dumps({"error": "e" * 500}))
    assert len(str(ei.value)) == 200


def test_odd_lines_mixed_into_real_http_stream(monkeypatch):
    lines = ['null', '{"message":{"content":"a"}}', '[1]', '12345678901234567890', '',
             '{"message":{"content":"b"},"done":false}', '"s"', '{"done":true}']
    state = {"closed": 0}

    class R:
        status_code = 200

        async def aiter_lines(self):
            for ln in lines:
                yield ln

    class Ctx:
        async def __aenter__(self):
            return R()

        async def __aexit__(self, *a):
            return False

    class Client:
        def __init__(self, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            state["closed"] += 1
            return False

        def stream(self, *a, **k):
            return Ctx()

    monkeypatch.setattr(chat_mod.httpx, "AsyncClient", Client)

    async def go():
        return [p async for p in chat_mod.ollama_pieces([])]
    assert asyncio.run(go()) == ["a", "b"]
    assert state["closed"] == 1


def test_ollama_pieces_closes_client_when_abandoned(monkeypatch):
    state = {"exited": 0, "client_exited": 0}

    class R:
        status_code = 200

        async def aiter_lines(self):
            for i in range(10):
                yield json.dumps({"message": {"content": str(i)}})

    class Ctx:
        async def __aenter__(self):
            return R()

        async def __aexit__(self, *a):
            state["exited"] += 1
            return False

    class Client:
        def __init__(self, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            state["client_exited"] += 1
            return False

        def stream(self, *a, **k):
            return Ctx()

    monkeypatch.setattr(chat_mod.httpx, "AsyncClient", Client)

    async def go():
        g = chat_mod.ollama_pieces([])
        await g.__anext__()
        await g.aclose()
        await g.aclose()
    asyncio.run(go())
    assert state == {"exited": 1, "client_exited": 1}


# ---- interruption marker -----------------------------------------------------------

@pytest.mark.parametrize("reason", ["héllo 世界 \U0001F600", "a\nb", "x" * 500, "]"])
def test_marker_wording_and_cap_for_stream_errors(reason):
    with pytest.raises(chat_mod.OllamaStreamError) as ei:
        chat_mod.piece_from_line(json.dumps({"error": reason}))
    note = chat_mod.interruption_note(ei.value)
    assert note.startswith("\n\n[Reply interrupted: Ollama returned an error: ")
    assert note.endswith("]")
    assert len(str(ei.value)) <= 200
    assert json.loads(json.dumps(note)) == note


def test_marker_for_http_errors_is_ascii_single_line():
    for e in (httpx.ConnectError("x"), httpx.ReadTimeout("x"), httpx.ReadError("x"), httpx.DecodingError("x")):
        note = chat_mod.interruption_note(e)
        assert note.startswith("\n\n[Reply interrupted: ") and note.endswith("]")
        assert note.isascii() and "\n" not in note.strip()


# ---- retrieve --------------------------------------------------------------------

class StoreStub:
    def __init__(self, has=True, hits=None, exc=None, where="query"):
        self.has, self.hits, self.exc, self.where = has, hits or [], exc, where

    def has_index(self, pid):
        if self.exc and self.where == "has_index":
            raise self.exc
        return self.has

    def query(self, pid, q, k):
        if self.exc and self.where == "query":
            raise self.exc
        return self.hits


@pytest.mark.parametrize("exc", [RuntimeError("x"), ValueError(), OSError("disk"), KeyError("k"),
                                 httpx.ConnectError("n"), MemoryError(), RecursionError(), AssertionError()])
@pytest.mark.parametrize("where", ["has_index", "query"])
def test_retrieve_exception_subclasses_are_failed(monkeypatch, exc, where):
    monkeypatch.setattr(chat_mod.rag_store, "store", StoreStub(exc=exc, where=where))
    assert asyncio.run(chat_mod.retrieve(1, "q")) == ("failed", [])


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), SystemExit(), asyncio.CancelledError()])
def test_retrieve_base_exceptions_propagate(monkeypatch, exc):
    monkeypatch.setattr(chat_mod.rag_store, "store", StoreStub(exc=exc))

    async def go():
        return await chat_mod.retrieve(1, "q")
    with pytest.raises(type(exc)):
        asyncio.run(go())


def test_retrieve_malformed_hit_is_failed(monkeypatch):
    monkeypatch.setattr(chat_mod.rag_store, "store", StoreStub(hits=[{"text": "no score"}]))
    assert asyncio.run(chat_mod.retrieve(1, "q")) == ("failed", [])


def test_retrieve_statuses(monkeypatch):
    assert asyncio.run(chat_mod.retrieve(None, "q")) == (None, [])
    assert asyncio.run(chat_mod.retrieve(0, "q")) == (None, [])
    monkeypatch.setattr(chat_mod.rag_store, "store", StoreStub(has=False))
    assert asyncio.run(chat_mod.retrieve(1, "q")) == ("empty", [])
    monkeypatch.setattr(chat_mod.rag_store, "store", StoreStub(hits=[]))
    assert asyncio.run(chat_mod.retrieve(1, "q")) == ("none", [])


# ---- title_from_message ---------------------------------------------------------------

@pytest.mark.parametrize("text", ["", "   ", "\n\t", "​", "\x00\x01"])
def test_title_nothing_printable_gets_default(text):
    assert sess.title_from_message(text) == "New chat"


@pytest.mark.parametrize("text", ["........", "?!?!", "---", "...", "!"])
def test_title_punctuation_only_is_kept(text):
    assert sess.title_from_message(text) == text


def test_title_emoji_only():
    assert sess.title_from_message("\U0001F600\U0001F600") == "\U0001F600\U0001F600"


def test_title_literal_new_chat_message():
    assert sess.title_from_message("New chat") == "New chat"


def test_title_length_boundaries():
    t60, t61 = "a " * 29 + "bb", "a " * 29 + "bbb"
    assert len(t60) == 60 and sess.title_from_message(t60) == t60
    out = sess.title_from_message(t61)
    assert len(t61) == 61 and len(out) <= 60 and out.endswith("…")
    out = sess.title_from_message("x" * 61)
    assert len(out) == 60 and out.endswith("…")


def test_title_never_empty_never_over_limit_fuzz():
    import random
    rnd = random.Random(7)
    alphabet = "ab .,;:-!…\n\U0001F600é"
    for _ in range(500):
        s = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(0, 150)))
        t = sess.title_from_message(s)
        assert t and len(t) <= 60 and "\n" not in t


def test_title_short_trailing_ellipsis_kept():
    assert sess.title_from_message("wait…") == "wait…"


def test_title_first_sentence_used():
    assert sess.title_from_message("How do I install this? Also other things.") == "How do I install this?"


# ---- failReply / static -----------------------------------------------------------------

JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")


def _fail_reply():
    s = JS.index("function failReply")
    return JS[s:JS.index("\nasync function stream", s)]


def test_failreply_appends_to_existing_draft_not_overwrites():
    b = _fail_reply()
    assert "q.value.trim()" in b
    assert "q.value.replace(/\\s+$/, '')" in b
    assert "+ text" in b


def test_failreply_refused_removes_both_bubbles_and_skips_retry():
    b = _fail_reply()
    refused = b[b.index("if (refused)"):b.index("out.classList.add('err')")]
    assert "out.remove()" in refused and "mine.remove()" in refused and "return;" in refused
    assert "send(" not in refused


def test_failreply_retry_removes_then_resends():
    b = _fail_reply()
    on = b.index("onclick")
    assert b.index("out.remove()", on) < b.index("send(text, false)", on)


# ---- repo-wide merge hygiene / compile ----------------------------------------------------

def _repo_files():
    r = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                       cwd=ROOT, capture_output=True)
    if r.returncode == 0 and r.stdout:
        return [ROOT / p for p in r.stdout.decode().split("\0") if p]
    return [p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts]


def test_no_conflict_markers_in_any_repo_file():
    bad = []
    for p in _repo_files():
        if not p.is_file() or ".git" in p.relative_to(ROOT).parts:
            continue
        if p.name.startswith("test_") and p.suffix == ".py":
            continue  # tests may mention the markers as strings
        try:
            data = p.read_bytes()
        except OSError:
            continue
        if b"\0" in data[:4096]:
            continue
        for n, ln in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
            if ln.startswith(("<<<<<<<", ">>>>>>>")) or ln.rstrip() == "=======":
                bad.append(f"{p.relative_to(ROOT)}:{n}")
    assert not bad, bad


def test_no_unmerged_index_entries():
    r = subprocess.run(["git", "ls-files", "-u"], cwd=ROOT, capture_output=True, text=True)
    if r.returncode == 0:
        assert r.stdout.strip() == ""


def test_compileall_clean():
    import compileall
    for target in ("rag.py", "app.py", "backend"):
        pth = ROOT / target
        ok = (compileall.compile_file(pth, quiet=2, force=True) if pth.is_file()
              else compileall.compile_dir(pth, quiet=2, force=True))
        assert ok, target
