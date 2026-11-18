"""User-feedback fixes: message limits, history/RAG honesty, rename and titles, clean Ollama
errors, health check, orphan cleanup, duplicate project names, UTF-8 file names, Stop (disconnect)."""
import json
import sqlite3
import sys
from pathlib import Path

import anyio
import httpx
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app as server  # noqa: E402
import rag  # noqa: E402
from backend import config, database, rag_store  # noqa: E402
from backend.ollama_errors import describe_ollama_error  # noqa: E402
from backend.routes import chat as chat_routes  # noqa: E402
from backend.routes import files as files_routes  # noqa: E402
from backend.routes import health  # noqa: E402
from backend.routes.sessions import title_from_message  # noqa: E402

real_installed_models = health.installed_models  # conftest stubs the module attribute for every test


def keyword_embedder(texts):
    """banana / quantum / anything else: three axes, so scores are exact and predictable."""
    return [[t.lower().count("banana"), t.lower().count("quantum"), 0.0 if ("banana" in t.lower() or "quantum" in t.lower()) else 1.0]
            for t in texts]


class FakeUpstream:
    """Stands in for httpx.AsyncClient talking to Ollama's streaming /api/chat."""

    def __init__(self):
        self.requests = []
        self.lines = [json.dumps({"message": {"content": "ok"}})]
        self.status = 200
        self.error_body = {"error": "x"}
        self.connect_error = None   # raised when the stream is opened
        self.mid_error = None       # raised after the lines were delivered
        self.endless = False        # deliver "tick" pieces forever
        self.closed = 0             # how many responses were closed

    def install(self, monkeypatch):
        upstream = self

        class Resp:
            @property
            def status_code(self):
                return upstream.status

            async def aread(self):
                return b""

            def raise_for_status(self):
                req = httpx.Request("POST", "http://x/api/chat")
                raise httpx.HTTPStatusError("bad", request=req, response=httpx.Response(
                    upstream.status, json=upstream.error_body, request=req))

            async def aiter_lines(self):
                for line in upstream.lines:
                    yield line
                if upstream.endless:
                    while True:
                        yield json.dumps({"message": {"content": "tick"}})
                        await anyio.sleep(0.001)
                if upstream.mid_error:
                    raise upstream.mid_error

            async def __aenter__(self):
                if upstream.connect_error:
                    raise upstream.connect_error
                return self

            async def __aexit__(self, *a):
                upstream.closed += 1
                return False

        class Client:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            def stream(self, method, url, json=None):
                upstream.requests.append(json)
                return Resp()

        monkeypatch.setattr(httpx, "AsyncClient", Client)
        return self


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "server.db")
    server.init_db()
    monkeypatch.setattr(
        rag_store, "store",
        rag.RagStore(tmp_path / "rag.db", tmp_path / "copies", keyword_embedder, model="fake"),
    )
    monkeypatch.setattr(files_routes, "UPLOADS", tmp_path / "uploads")
    (tmp_path / "uploads").mkdir()
    return TestClient(server.app)


@pytest.fixture
def ollama(monkeypatch):
    return FakeUpstream().install(monkeypatch)


def stored(sid=None):
    with database.db() as c:
        rows = c.execute(
            "SELECT role, content FROM messages" + (" WHERE session_id = ?" if sid else "") + " ORDER BY id",
            (sid,) if sid else (),
        ).fetchall()
    return [tuple(r) for r in rows]


def new_chat(client, project_id=None):
    url = f"/api/projects/{project_id}/sessions" if project_id else "/api/sessions"
    return client.post(url).json()["id"]


def new_project(client, name="p", **files):
    pid = client.post("/api/projects", json={"name": name}).json()["id"]
    if files:
        r = client.post(f"/api/projects/{pid}/files", files=[("files", (n, d)) for n, d in files.items()])
        assert all(x["result"] == "indexed" for x in r.json()["results"])
    return pid


def system_text(request):
    return "\n".join(m["content"] for m in request["messages"] if m["role"] == "system")


# ---- 1. message length cap ---------------------------------------------------------------

def test_default_limit_is_20000():
    assert config.MAX_MESSAGE_CHARS == 20_000


def test_too_long_message_is_413_with_clear_detail_and_nothing_is_stored(client, ollama):
    sid = new_chat(client)
    r = client.post("/api/chat", json={"session_id": sid, "content": "x" * 20_001})
    assert r.status_code == 413
    assert r.json()["detail"] == "Message too long (20001 characters, max 20000)"
    assert stored() == [] and ollama.requests == []
    assert client.get(f"/api/sessions/{sid}").json()["title"] == "New chat"


def test_message_of_exactly_the_limit_is_accepted(client, ollama):
    sid = new_chat(client)
    assert client.post("/api/chat", json={"session_id": sid, "content": "x" * 20_000}).status_code == 200


def test_limit_follows_the_config(client, ollama, monkeypatch):
    monkeypatch.setattr(config, "MAX_MESSAGE_CHARS", 10)
    sid = new_chat(client)
    r = client.post("/api/chat", json={"session_id": sid, "content": "y" * 11})
    assert r.status_code == 413 and r.json()["detail"] == "Message too long (11 characters, max 10)"


@pytest.mark.parametrize("content", ["", " ", "   ", "\n\t \r\n", "  "])
def test_blank_message_is_400_and_not_stored(client, ollama, content):
    sid = new_chat(client)
    r = client.post("/api/chat", json={"session_id": sid, "content": content})
    assert r.status_code == 400 and "empty" in r.json()["detail"].lower()
    assert stored() == [] and ollama.requests == []


def test_content_is_stored_as_sent_not_stripped(client, ollama):
    sid = new_chat(client)
    client.post("/api/chat", json={"session_id": sid, "content": "  keep   the\n  spacing \n"})
    assert stored(sid)[0] == ("user", "  keep   the\n  spacing \n")


# ---- 1b. history trimmed header ----------------------------------------------------------

def test_history_trimmed_header_counts_messages_left_out(client, ollama):
    sid = new_chat(client)
    with database.db() as c:
        for i in range(3):
            c.execute("INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
                      (sid, "user" if i % 2 == 0 else "assistant", "m" * 10_000, i))
    r = client.post("/api/chat", json={"session_id": sid, "content": "latest"})
    assert r.headers["X-History-Trimmed"] == "1"
    assert len([m for m in ollama.requests[0]["messages"] if m["role"] != "system"]) == 3


def test_history_trimmed_header_absent_when_everything_fits(client, ollama):
    sid = new_chat(client)
    client.post("/api/chat", json={"session_id": sid, "content": "one"})
    r = client.post("/api/chat", json={"session_id": sid, "content": "two"})
    assert "X-History-Trimmed" not in r.headers


def test_history_trimmed_header_name_matches_the_constant():
    assert chat_routes.TRIMMED_HEADER == "X-History-Trimmed"


# ---- 2. relevance threshold and RAG status ----------------------------------------------

@pytest.fixture
def threshold(monkeypatch):
    monkeypatch.setattr(config, "RAG_MIN_SCORE", 0.5)


def ask(client, sid, content):
    return client.post("/api/chat", json={"session_id": sid, "content": content})


def test_default_threshold_is_a_cosine_in_the_measured_gap():
    assert 0.4 < config.RAG_MIN_SCORE < 0.6 or config.RAG_MIN_SCORE == -1.0  # conftest switches it off


def test_status_used_lists_only_files_that_were_injected(client, ollama, threshold):
    pid = new_project(client, **{"a.md": b"banana smoothie", "b.md": b"quantum entanglement"})
    r = ask(client, new_chat(client, pid), "banana")
    assert r.headers["X-RAG-Status"] == "used"
    assert json.loads(r.headers["X-RAG-Sources"]) == ["a.md"]
    sys_text = system_text(ollama.requests[0])
    assert "banana smoothie" in sys_text and "quantum" not in sys_text


def test_chunks_below_the_threshold_are_dropped_and_at_or_above_kept(client, ollama, threshold):
    pid = new_project(client, **{"a.md": b"banana smoothie"})
    sid = new_chat(client, pid)
    # query vector [1, 2, 0] scores 1/sqrt(5) = 0.447 against the banana chunk; [2, 1, 0] scores 0.894
    assert ask(client, sid, "banana quantum quantum").headers["X-RAG-Status"] == "none"
    assert ask(client, sid, "banana banana quantum").headers["X-RAG-Status"] == "used"


def test_status_none_injects_nothing_and_adds_the_hint(client, ollama, threshold):
    pid = new_project(client, **{"a.md": b"banana smoothie"})
    r = ask(client, new_chat(client, pid), "what is the weather")
    assert r.headers["X-RAG-Status"] == "none" and json.loads(r.headers["X-RAG-Sources"]) == []
    msgs = ollama.requests[0]["messages"]
    assert "<excerpts>" not in system_text(ollama.requests[0])
    assert msgs[0]["role"] == "system" and "no relevant excerpts" in msgs[0]["content"]
    assert msgs[-1] == {"role": "user", "content": "what is the weather"}  # hint is ahead of the history


def test_status_empty_for_a_project_without_files(client, ollama, threshold):
    pid = new_project(client)
    r = ask(client, new_chat(client, pid), "anything")
    assert r.headers["X-RAG-Status"] == "empty"
    assert ollama.requests[0]["messages"][0]["role"] == "system"
    assert "no indexed files" in system_text(ollama.requests[0])


@pytest.mark.parametrize("exc", [
    httpx.ConnectError("down"), httpx.ReadTimeout("slow"), rag.ReindexRequired("old"),
    rag.EmbeddingError("bad"), sqlite3.OperationalError("locked"), ValueError("x"),
])
def test_status_failed_when_retrieval_raises_and_chat_still_answers(client, ollama, threshold, monkeypatch, exc):
    pid = new_project(client, **{"a.md": b"banana smoothie"})

    def boom(*a, **k):
        raise exc
    monkeypatch.setattr(rag_store.store, "query", boom)
    r = ask(client, new_chat(client, pid), "banana")
    assert r.status_code == 200 and r.text == "ok" and r.headers["X-RAG-Status"] == "failed"
    assert "Searching the project files failed" in system_text(ollama.requests[0])


def test_status_failed_when_has_index_raises(client, ollama, monkeypatch):
    pid = new_project(client, **{"a.md": b"banana"})

    def boom(*a):
        raise sqlite3.OperationalError("locked")
    monkeypatch.setattr(rag_store.store, "has_index", boom)
    assert ask(client, new_chat(client, pid), "banana").headers["X-RAG-Status"] == "failed"


def test_normal_chat_has_no_status_header_and_no_system_message(client, ollama, threshold):
    r = ask(client, new_chat(client), "banana")
    assert "X-RAG-Status" not in r.headers and json.loads(r.headers["X-RAG-Sources"]) == []
    assert [m["role"] for m in ollama.requests[0]["messages"]] == ["user"]


def test_status_used_has_no_hint(client, ollama, threshold):
    pid = new_project(client, **{"a.md": b"banana smoothie"})
    ask(client, new_chat(client, pid), "banana")
    assert "no relevant excerpts" not in system_text(ollama.requests[0])


def test_status_header_name_matches_the_constant():
    assert chat_routes.STATUS_HEADER == "X-RAG-Status"


@pytest.mark.anyio
async def test_retrieve_applies_the_threshold(client, threshold):
    pid = new_project(client, **{"a.md": b"banana smoothie"})
    assert (await chat_routes.retrieve(pid, "banana"))[0] == "used"
    assert (await chat_routes.retrieve(pid, "weather"))[0] == "none"
    assert await chat_routes.retrieve(None, "banana") == (None, [])


# ---- 3. rename and titles ---------------------------------------------------------------

def test_rename_a_normal_chat(client):
    sid = new_chat(client)
    r = client.put(f"/api/sessions/{sid}", json={"title": "  My notes  "})
    assert r.status_code == 200 and r.json() == {"id": sid, "title": "My notes"}
    assert [s["title"] for s in client.get("/api/sessions").json()] == ["My notes"]
    assert client.get(f"/api/sessions/{sid}").json()["title"] == "My notes"


def test_rename_a_session_inside_a_project(client):
    pid = new_project(client)
    sid = new_chat(client, pid)
    assert client.put(f"/api/sessions/{sid}", json={"title": "Chapter 1"}).status_code == 200
    assert client.get(f"/api/projects/{pid}").json()["sessions"][0]["title"] == "Chapter 1"
    assert client.get("/api/sessions").json() == []  # still not a normal chat


@pytest.mark.parametrize("title", ["", "   ", "\n\t"])
def test_rename_blank_is_400(client, title):
    sid = new_chat(client)
    assert client.put(f"/api/sessions/{sid}", json={"title": title}).status_code == 400
    assert client.get(f"/api/sessions/{sid}").json()["title"] == "New chat"


def test_rename_length_limits(client):
    sid = new_chat(client)
    assert client.put(f"/api/sessions/{sid}", json={"title": "t" * 200}).status_code == 200
    assert client.put(f"/api/sessions/{sid}", json={"title": "t" * 201}).status_code == 400
    assert client.put(f"/api/sessions/{sid}", json={"title": "  " + "t" * 200 + "  "}).status_code == 200


def test_rename_unknown_session_is_404_and_bad_body_422(client):
    assert client.put("/api/sessions/999", json={"title": "x"}).status_code == 404
    sid = new_chat(client)
    assert client.put(f"/api/sessions/{sid}", json={}).status_code == 422


def test_rename_unicode_title(client):
    sid = new_chat(client)
    assert client.put(f"/api/sessions/{sid}", json={"title": "日本語 \U0001f600"}).json()["title"] == "日本語 \U0001f600"


@pytest.mark.parametrize("text, expected", [
    ("How do I reverse a list in Python?", "How do I reverse a list in Python?"),
    ("Fix this bug. Here is the long stack trace that follows and keeps going on and on and on", "Fix this bug"),
    ("hello\n\n   world", "hello world"),
    ("Use v3.5 of the library now please", "Use v3.5 of the library now please"),
    ("Hi. Can you explain what a closure is?", "Hi. Can you explain what a closure is?"),  # too-short first sentence
])
def test_title_from_message_short_cases(text, expected):
    assert title_from_message(text) == expected


def test_long_title_is_cut_at_a_word_boundary_with_an_ellipsis():
    text = "Explain the difference between TCP and UDP in networking protocols, please be very detailed"
    title = title_from_message(text)
    assert title == "Explain the difference between TCP and UDP in networking…"
    assert len(title) <= 60 and text.startswith(title[:-1]) and text[len(title) - 1] == " "


@pytest.mark.parametrize("n", range(45, 80))
def test_long_titles_never_end_mid_word(n):
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    text = " ".join(words * 6)[: n + 30]
    title = title_from_message(text)
    if len(text) > 60:
        assert title.endswith("…") and len(title) <= 60
        cut = title[:-1]
        assert text.startswith(cut) and (text[len(cut)] == " " or len(cut) == 0)
    else:
        assert title == text


def test_unbreakable_long_text_is_hard_cut_with_an_ellipsis():
    title = title_from_message("x" * 200)
    assert title == "x" * 59 + "…"


def test_chat_names_session_from_first_message_only(client, ollama):
    sid = new_chat(client)
    ask(client, sid, "What is a monad? It confuses me")
    ask(client, sid, "Something else entirely")
    assert client.get(f"/api/sessions/{sid}").json()["title"] == "What is a monad?"


def test_a_renamed_chat_keeps_its_name_when_the_first_message_arrives(client, ollama):
    sid = new_chat(client)
    client.put(f"/api/sessions/{sid}", json={"title": "Mine"})
    ask(client, sid, "first message")
    assert client.get(f"/api/sessions/{sid}").json()["title"] == "Mine"


def test_existing_titles_are_untouched(client, ollama):
    sid = client.post("/api/sessions", json={"title": "Old title"}).json()["id"]
    ask(client, sid, "hello there")
    assert client.get(f"/api/sessions/{sid}").json()["title"] == "Old title"


# ---- 4. error messages ---------------------------------------------------------------------

def status_error(code, body=None):
    req = httpx.Request("POST", "http://x/api/embed")
    return httpx.HTTPStatusError("boom", request=req, response=httpx.Response(code, json=body or {}, request=req))


@pytest.mark.parametrize("role, label", [("embed", "embedding"), ("chat", "chat"), ("vision", "vision")])
def test_404_means_model_not_installed(role, label):
    msg = describe_ollama_error(status_error(404, {"error": "model not found"}), "some-model:1b", role)
    assert msg == f"The {label} model 'some-model:1b' isn't installed. Run: ollama pull some-model:1b"


@pytest.mark.parametrize("exc", [httpx.ConnectError("All connection attempts failed"), httpx.ConnectTimeout("t")])
def test_connection_errors_say_ollama_is_unreachable(exc):
    assert describe_ollama_error(exc, "m") == f"Can't reach Ollama at {config.OLLAMA}. Is it running?"


@pytest.mark.parametrize("exc", [httpx.ReadTimeout("t"), httpx.WriteTimeout("t"), httpx.PoolTimeout("t")])
def test_timeouts(exc):
    assert describe_ollama_error(exc, "m") == "Ollama took too long to answer"


def test_other_statuses_use_ollamas_own_short_message():
    assert describe_ollama_error(status_error(500, {"error": "out of memory"}), "m") == \
        "Ollama returned an error (500): out of memory"
    assert describe_ollama_error(status_error(500), "m") == "Ollama returned an error (500)."
    long = describe_ollama_error(status_error(500, {"error": "z" * 500}), "m")
    assert "zzz" not in long


def test_no_message_leaks_raw_httpx_text():
    for exc in [httpx.ConnectError("All connection attempts failed"), httpx.ReadTimeout("x"),
                status_error(404), status_error(500), httpx.RemoteProtocolError("peer closed"),
                httpx.ReadError("boom")]:
        msg = describe_ollama_error(exc, "m")
        assert "httpx" not in msg and "mozilla" not in msg and "mdn" not in msg.lower()
        assert "All connection attempts failed" not in msg and "boom" not in msg


def test_upload_embed_model_missing_names_the_embedding_model_only(client, monkeypatch):
    monkeypatch.setattr(rag_store.store, "embedder", rag.OllamaEmbedder("http://x", config.EMBED_MODEL))
    req = httpx.Request("POST", "http://x/api/embed")
    monkeypatch.setattr(httpx, "post", lambda *a, **k: httpx.Response(404, json={"error": "nope"}, request=req))
    pid = new_project(client)
    res = client.post(f"/api/projects/{pid}/files", files={"files": ("n.txt", b"text")}).json()["results"][0]
    assert res["status"] == 502
    assert res["detail"] == (f"The embedding model '{config.EMBED_MODEL}' isn't installed. "
                             f"Run: ollama pull {config.EMBED_MODEL}")
    assert config.VISION_MODEL not in res["detail"]


@pytest.mark.parametrize("exc, expected", [
    (httpx.ConnectError("refused"), f"Can't reach Ollama at {config.OLLAMA}. Is it running?"),
    (httpx.ReadTimeout("slow"), "Ollama took too long to answer"),
])
def test_upload_connection_and_timeout_messages(client, monkeypatch, exc, expected):
    monkeypatch.setattr(rag_store.store, "embedder", rag.OllamaEmbedder("http://x", config.EMBED_MODEL))

    def boom(*a, **k):
        raise exc
    monkeypatch.setattr(httpx, "post", boom)
    pid = new_project(client)
    res = client.post(f"/api/projects/{pid}/files", files={"files": ("n.txt", b"text")}).json()["results"][0]
    assert res["detail"] == expected


def test_debug_rag_endpoint_uses_the_same_messages(client, monkeypatch):
    pid = new_project(client, **{"a.md": b"banana"})
    monkeypatch.setattr(rag_store.store, "query_embedder", None)
    monkeypatch.setattr(rag_store.store, "embedder", rag.OllamaEmbedder("http://x", config.EMBED_MODEL))
    monkeypatch.setattr(httpx, "post", lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("x")))
    r = client.get(f"/api/projects/{pid}/rag", params={"q": "banana"})
    assert r.status_code == 502 and r.json()["detail"].startswith("Can't reach Ollama")


def test_chat_model_missing_is_a_502_with_the_pull_command(client, ollama):
    ollama.status = 404
    sid = new_chat(client)
    r = ask(client, sid, "hi")
    assert r.status_code == 502
    assert r.json()["detail"] == f"The chat model '{config.MODEL}' isn't installed. Run: ollama pull {config.MODEL}"


def test_chat_with_ollama_down_is_a_502_with_a_readable_detail(client, ollama):
    ollama.connect_error = httpx.ConnectError("All connection attempts failed")
    r = ask(client, new_chat(client), "hi")
    assert r.status_code == 502
    assert r.json()["detail"] == f"Can't reach Ollama at {config.OLLAMA}. Is it running?"


def test_chat_timeout_message(client, ollama):
    ollama.connect_error = httpx.ReadTimeout("slow")
    assert ask(client, new_chat(client), "hi").json()["detail"] == "Ollama took too long to answer"


def test_chat_error_line_from_ollama_is_reported(client, ollama):
    ollama.lines = [json.dumps({"error": "model requires more memory"})]
    r = ask(client, new_chat(client), "hi")
    assert r.status_code == 502 and "model requires more memory" in r.json()["detail"]


def test_reindex_conflict_message_is_actionable(client):
    pid = new_project(client, **{"a.md": b"banana"})
    with sqlite3.connect(rag_store.store.db_path) as c:
        c.execute("UPDATE rag_chunks SET model = 'older'")
    r = client.get(f"/api/projects/{pid}/rag", params={"q": "banana"})
    assert r.status_code == 409 and "Upload the project's files again" in r.json()["detail"]


# ---- 5. orphan message cleanup -----------------------------------------------------------

@pytest.mark.parametrize("setup", ["down", "no_model", "empty", "error_line"])
def test_failed_reply_leaves_no_unanswered_message(client, ollama, setup):
    sid = new_chat(client)
    if setup == "down":
        ollama.connect_error = httpx.ConnectError("x")
    elif setup == "no_model":
        ollama.status = 404
    elif setup == "empty":
        ollama.lines = []
    else:
        ollama.lines = [json.dumps({"error": "boom"})]
    assert ask(client, sid, "will fail").status_code == 502
    assert stored(sid) == []
    assert client.get(f"/api/sessions/{sid}").json()["messages"] == []


def test_cleanup_removes_only_the_failed_message(client, ollama):
    sid = new_chat(client)
    other = new_chat(client)
    ask(client, sid, "first")
    ask(client, other, "elsewhere")
    ollama.connect_error = httpx.ConnectError("x")
    ask(client, sid, "second fails")
    assert stored(sid) == [("user", "first"), ("assistant", "ok")]
    assert stored(other) == [("user", "elsewhere"), ("assistant", "ok")]


def test_failed_message_is_not_resent_as_history(client, ollama):
    sid = new_chat(client)
    ollama.connect_error = httpx.ConnectError("x")
    ask(client, sid, "same text")
    ollama.connect_error = None
    r = ask(client, sid, "same text")  # Retry
    assert r.status_code == 200
    assert [m["content"] for m in ollama.requests[-1]["messages"]] == ["same text"]
    assert stored(sid) == [("user", "same text"), ("assistant", "ok")]


def test_unexpected_error_before_the_reply_also_removes_the_message(client, ollama, monkeypatch):
    async def boom(*a):
        raise RuntimeError("bug")
    monkeypatch.setattr(chat_routes, "retrieve", boom)
    sid = new_chat(client)
    with pytest.raises(RuntimeError):
        ask(client, sid, "hi")
    assert stored(sid) == []


def test_partial_reply_before_ollama_drops_is_kept(client, ollama):
    ollama.lines = [json.dumps({"message": {"content": "part"}})]
    ollama.mid_error = httpx.ReadError("dropped")
    sid = new_chat(client)
    r = ask(client, sid, "q")
    assert r.status_code == 200 and r.text == "part"
    assert stored(sid) == [("user", "q"), ("assistant", "part")]


def test_first_message_title_is_reapplied_on_retry(client, ollama):
    sid = new_chat(client)
    ollama.connect_error = httpx.ConnectError("x")
    ask(client, sid, "Explain monads. Please.")
    ollama.connect_error = None
    ask(client, sid, "Explain monads. Please.")
    assert client.get(f"/api/sessions/{sid}").json()["title"] == "Explain monads"


# ---- 4b. health ----------------------------------------------------------------------------

@pytest.fixture
def tags(monkeypatch):
    """Controls what Ollama 'has installed'; counts probes. None means unreachable."""
    state = {"models": {config.MODEL, config.EMBED_MODEL + ":latest", config.VISION_MODEL}, "calls": 0}

    async def installed():
        state["calls"] += 1
        return state["models"]

    monkeypatch.setattr(health, "installed_models", installed)
    monkeypatch.setitem(health._cache, "at", None)
    return state


def test_health_keeps_the_old_fields_and_adds_the_new(client, tags):
    h = client.get("/api/health").json()
    assert h["ok"] is True and h["model"] == config.MODEL
    assert h["status"] == "ok" and h["ollama"] is True
    assert h["models"] == {"chat": True, "embed": True, "vision": True}


def test_health_is_degraded_but_ok_when_ollama_is_unreachable(client, tags):
    tags["models"] = None
    h = client.get("/api/health").json()
    assert h["ok"] is True and h["status"] == "degraded" and h["ollama"] is False
    assert h["models"] == {"chat": False, "embed": False, "vision": False}


@pytest.mark.parametrize("missing, key", [("chat", "chat"), ("embed", "embed")])
def test_health_degraded_when_a_needed_model_is_missing(client, tags, missing, key):
    wanted = config.MODEL if missing == "chat" else config.EMBED_MODEL + ":latest"
    tags["models"] = tags["models"] - {wanted}
    h = client.get("/api/health").json()
    assert h["status"] == "degraded" and h["ollama"] is True and h["models"][key] is False


def test_missing_vision_model_is_reported_but_does_not_degrade(client, tags):
    tags["models"] = tags["models"] - {config.VISION_MODEL}
    h = client.get("/api/health").json()
    assert h["models"]["vision"] is False and h["status"] == "ok"


def test_vision_is_null_when_disabled(client, tags, monkeypatch):
    monkeypatch.setattr(health, "VISION_MODEL", "")
    assert client.get("/api/health").json()["models"]["vision"] is None


def test_health_is_cached_for_a_few_seconds(client, tags, monkeypatch):
    client.get("/api/health")
    client.get("/api/health")
    assert tags["calls"] == 1
    monkeypatch.setattr(config, "HEALTH_CACHE_SECONDS", 0)
    client.get("/api/health")
    assert tags["calls"] == 2


def test_health_cache_ttl_is_about_five_seconds():
    assert config.HEALTH_CACHE_SECONDS == 5 and config.HEALTH_TIMEOUT == 2


@pytest.mark.parametrize("wanted, installed, expected", [
    ("nomic-embed-text", {"nomic-embed-text:latest"}, True),
    ("qwen2.5-coder:7b", {"qwen2.5-coder:7b"}, True),
    ("qwen2.5-coder:7b", {"qwen2.5-coder:3b"}, False),
    ("nomic-embed-text", {"nomic-embed-text-v2:latest"}, False),
    ("llama3", set(), False),
])
def test_has_model_matches_ollama_names(wanted, installed, expected):
    assert health.has_model(installed, wanted) is expected


@pytest.mark.anyio
async def test_installed_models_reads_api_tags(monkeypatch):
    class Client:
        def __init__(self, *a, **kw):
            self.timeout = kw.get("timeout")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url):
            assert url.endswith("/api/tags") and self.timeout == config.HEALTH_TIMEOUT
            return httpx.Response(200, json={"models": [{"name": "a:1"}, {"name": "b:latest"}, {}]},
                                  request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    assert await real_installed_models() == {"a:1", "b:latest"}


@pytest.mark.anyio
@pytest.mark.parametrize("failure", [httpx.ConnectError("x"), httpx.ReadTimeout("x")])
async def test_installed_models_is_none_when_unreachable(monkeypatch, failure):
    class Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url):
            raise failure

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    assert await real_installed_models() is None


@pytest.mark.anyio
async def test_installed_models_is_none_for_garbage_answers(monkeypatch):
    class Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url):
            return httpx.Response(200, text="not json", request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    assert await real_installed_models() is None


# ---- 5b. duplicate project names ---------------------------------------------------------

def test_duplicate_project_name_is_409(client):
    assert client.post("/api/projects", json={"name": "Thesis"}).status_code == 200
    for name in ["Thesis", "thesis", "  THESIS  ", "tHeSiS"]:
        r = client.post("/api/projects", json={"name": name})
        assert r.status_code == 409
        assert r.json()["detail"] == f"A project named '{name.strip()}' already exists"
    assert len(client.get("/api/projects").json()) == 1


def test_non_ascii_names_compare_case_insensitively(client):
    client.post("/api/projects", json={"name": "Straße"})
    assert client.post("/api/projects", json={"name": "STRASSE"}).status_code == 409
    client.post("/api/projects", json={"name": "École"})
    assert client.post("/api/projects", json={"name": "école"}).status_code == 409


def test_different_names_are_fine(client):
    assert client.post("/api/projects", json={"name": "A"}).status_code == 200
    assert client.post("/api/projects", json={"name": "A 2"}).status_code == 200


def test_deleting_frees_the_name(client):
    pid = client.post("/api/projects", json={"name": "Once"}).json()["id"]
    client.delete(f"/api/projects/{pid}")
    assert client.post("/api/projects", json={"name": "once"}).status_code == 200


def test_databases_that_already_contain_duplicates_keep_working(client):
    with database.db() as c:
        c.execute("INSERT INTO projects (name, path) VALUES ('Dup', '')")
        c.execute("INSERT INTO projects (name, path) VALUES ('dup', '')")
    server.init_db()  # no migration trips over them
    assert [p["name"] for p in client.get("/api/projects").json()] == ["Dup", "dup"]
    assert client.post("/api/projects", json={"name": "DUP"}).status_code == 409
    assert client.post("/api/projects", json={"name": "fresh"}).status_code == 200


# ---- 5c. UTF-8 file names ------------------------------------------------------------------

NAMES = ["héllo wörld.txt", "日本語.txt", "smile\U0001f600.txt", "مرحبا.md"]


def raw_multipart(field, filename_bytes, content=b"hello"):
    body = (b'--B\r\nContent-Disposition: form-data; name="' + field.encode() + b'"; filename="'
            + filename_bytes + b'"\r\nContent-Type: text/plain\r\n\r\n' + content + b"\r\n--B--\r\n")
    return {"content": body, "headers": {"content-type": "multipart/form-data; boundary=B"}}


@pytest.mark.parametrize("name", NAMES)
def test_project_upload_keeps_utf8_names(client, name):
    pid = new_project(client)
    res = client.post(f"/api/projects/{pid}/files", files={"files": (name, b"some text")}).json()["results"]
    assert res == [{"path": name, "result": "indexed"}]
    assert [f["path"] for f in client.get(f"/api/projects/{pid}/files").json()] == [name]
    assert (rag_store.store.files_dir / str(pid) / name).read_bytes() == b"some text"


@pytest.mark.parametrize("name", NAMES)
def test_project_upload_with_browser_style_raw_utf8_bytes(client, name):
    pid = new_project(client)
    r = client.post(f"/api/projects/{pid}/files", **raw_multipart("files", name.encode("utf-8")))
    assert r.json()["results"][0]["path"] == name


def test_project_upload_with_a_latin1_encoded_name_from_an_older_client(client):
    pid = new_project(client)
    r = client.post(f"/api/projects/{pid}/files", **raw_multipart("files", "héllo.txt".encode("latin-1")))
    assert r.json()["results"][0]["path"] == "héllo.txt"


@pytest.mark.parametrize("name", NAMES)
def test_file_transfer_keeps_utf8_names(client, name):
    r = client.post("/api/upload", files={"file": (name, b"data")})
    assert r.json()["name"] == name
    assert [f["name"] for f in client.get("/api/files").json()] == [name]
    assert client.get("/api/files/" + httpx.URL("/").copy_with(path="/" + name).path.lstrip("/")).content == b"data"
    assert (files_routes.UPLOADS / name).is_file()


@pytest.mark.parametrize("name", NAMES)
def test_file_transfer_with_browser_style_raw_utf8_bytes(client, name):
    r = client.post("/api/upload", **raw_multipart("file", name.encode("utf-8")))
    assert r.json()["name"] == name


def test_duplicate_utf8_upload_gets_a_numbered_name(client):
    client.post("/api/upload", files={"file": ("café.txt", b"1")})
    assert client.post("/api/upload", files={"file": ("café.txt", b"2")}).json()["name"] == "café_1.txt"


def test_json_responses_are_utf8_so_a_cp1252_console_shows_the_mojibake_not_the_server(client):
    """The reported 'hÃ©llo' is the UTF-8 bytes read with the wrong codec by the client."""
    r = client.post("/api/upload", files={"file": ("héllo.txt", b"x")})
    assert "héllo".encode("utf-8") in r.content
    assert r.content.decode("cp1252", errors="replace") != r.content.decode("utf-8")


def test_sources_header_survives_non_ascii_file_names(client, ollama, monkeypatch):
    monkeypatch.setattr(config, "RAG_MIN_SCORE", 0.5)
    pid = new_project(client, **{"café 日本.md": b"banana smoothie"})
    r = ask(client, new_chat(client, pid), "banana")
    assert json.loads(r.headers["X-RAG-Sources"]) == ["café 日本.md"]


# ---- 6. Stop: the client goes away mid-reply ---------------------------------------------

async def open_stream(sid, content="question"):
    resp = await chat_routes.chat(chat_routes.ChatIn(session_id=sid, content=content))
    return resp, resp.body_iterator


@pytest.mark.anyio
async def test_closing_the_body_iterator_stores_the_partial_reply_once_and_closes_upstream(client, ollama):
    ollama.endless = True
    ollama.lines = [json.dumps({"message": {"content": "par"}}), json.dumps({"message": {"content": "tial"}})]
    sid = new_chat(client)
    resp, it = await open_stream(sid)
    got = [await it.__anext__(), await it.__anext__()]
    assert got == ["par", "tial"]
    await it.aclose()
    assert stored(sid)[0] == ("user", "question")
    assert [r for r in stored(sid) if r[0] == "assistant"] == [("assistant", "partial")]
    assert ollama.closed == 1  # the upstream request was closed, so Ollama stops generating
    await it.aclose()  # closing again changes nothing
    assert len(stored(sid)) == 2


@pytest.mark.anyio
async def test_cancelling_the_streaming_task_stores_the_partial_reply(client, ollama):
    """What Starlette does on a client disconnect: cancel the task iterating the body."""
    ollama.endless = True
    ollama.lines = []
    sid = new_chat(client)
    resp, it = await open_stream(sid)
    seen = []

    async def consume():
        async for piece in it:
            seen.append(piece)

    async with anyio.create_task_group() as tg:
        tg.start_soon(consume)
        while len(seen) < 3:
            await anyio.sleep(0.005)
        tg.cancel_scope.cancel()
    replies = [r for r in stored(sid) if r[0] == "assistant"]
    assert len(replies) == 1
    assert replies[0][1] == "tick" * len(seen) or replies[0][1] == "tick" * (len(seen) + 1)
    assert ollama.closed == 1


@pytest.mark.anyio
async def test_stopping_after_the_first_piece_keeps_that_piece(client, ollama):
    ollama.endless = True
    ollama.lines = []
    sid = new_chat(client)
    resp, it = await open_stream(sid)
    assert await it.__anext__() == "tick"
    await it.aclose()
    assert [r for r in stored(sid) if r[0] == "assistant"] == [("assistant", "tick")]


@pytest.mark.anyio
async def test_a_completed_stream_stores_the_reply_once(client, ollama):
    ollama.lines = [json.dumps({"message": {"content": "a"}}), json.dumps({"message": {"content": "b"}})]
    sid = new_chat(client)
    resp, it = await open_stream(sid)
    assert [p async for p in it] == ["a", "b"]
    assert stored(sid) == [("user", "question"), ("assistant", "ab")]
    assert ollama.closed == 1


@pytest.mark.anyio
async def test_disconnect_after_the_session_was_deleted_stores_nothing(client, ollama):
    ollama.endless = True
    sid = new_chat(client)
    resp, it = await open_stream(sid)
    await it.__anext__()
    client.delete(f"/api/sessions/{sid}")
    await it.aclose()
    assert stored(sid) == []
