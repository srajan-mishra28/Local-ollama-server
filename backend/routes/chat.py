"""Chat: prompt assembly (project RAG excerpts + windowed history) and the streamed Ollama reply."""

import contextlib
import json
import time

import anyio
import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response, StreamingResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel

import rag
from backend import config, rag_store
from backend.config import CHAT_READ_TIMEOUT, MODEL, OLLAMA, RAG_MAX_EXCERPT_CHARS, RAG_TOP_K
from backend.database import db
from backend.ollama_errors import describe_ollama_error
from backend.prompt import build_retrieval_query, fit_history
from backend.routes.sessions import DEFAULT_TITLE, title_from_message

router = APIRouter()

SOURCES_HEADER = "X-RAG-Sources"  # JSON list of the file names whose excerpts were sent
STATUS_HEADER = "X-RAG-Status"  # used | none | empty | failed; absent for a normal chat
TRIMMED_HEADER = "X-History-Trimmed"  # how many older messages were left out of the prompt

# Told to the model (as a system message, ahead of the history) when a project
# chat got no excerpts, so it says so instead of inventing what the files hold.
RAG_HINTS = {
    "none": (
        "The project's files returned no relevant excerpts for this question. If you cannot "
        "answer from the conversation itself, say that the project files don't cover it "
        "instead of guessing."
    ),
    "empty": (
        "This project has no indexed files yet, so there are no excerpts. If you cannot "
        "answer from the conversation itself, say that the project has no files instead "
        "of guessing."
    ),
    "failed": (
        "Searching the project files failed, so no excerpts are available. Say so if the "
        "answer needs them."
    ),
}


class ChatIn(BaseModel):
    session_id: int
    content: str


class OllamaStreamError(Exception):
    """Ollama sent an {"error": ...} line instead of a reply."""


class ClientGone(Exception):
    """The browser disconnected (Stop, closed tab) before the first reply text arrived."""


DISCONNECT_POLL_SECONDS = 0.25  # how often to look for a vanished client while waiting for the first token


async def retrieve(project_id: int | None, query: str) -> tuple[str | None, list[dict]]:
    """(status, hits): the excerpts for the prompt (best first, within the size budget).

    ``status`` is None for a normal chat, else 'used' (excerpts found), 'none'
    (indexed, but nothing scored >= RAG_MIN_SCORE), 'empty' (nothing indexed) or
    'failed'. Retrieval is best-effort: no failure here (embedder down, stale
    index, corrupt data, database trouble) may break plain chat, it is reported
    as 'failed'. Cancellation is not an Exception and still propagates.
    """
    if not project_id:
        return None, []
    try:
        if not await run_in_threadpool(rag_store.store.has_index, project_id):
            return "empty", []
        hits = await run_in_threadpool(rag_store.store.query, project_id, query, RAG_TOP_K)
        relevant = [h for h in hits if h["score"] >= config.RAG_MIN_SCORE]
        hits = rag.fit_hits(relevant, RAG_MAX_EXCERPT_CHARS)
    except Exception:  # noqa: BLE001 - deliberately broad, see the docstring
        return "failed", []
    return ("used" if hits else "none"), hits


async def retrieve_hits(project_id: int | None, query: str) -> list[dict]:
    return (await retrieve(project_id, query))[1]


async def retrieve_context(project_id: int | None, query: str) -> str:
    """``retrieve_hits`` rendered as a system-message block ('' if none)."""
    return rag.format_hits(await retrieve_hits(project_id, query))


async def ollama_pieces(messages: list[dict]):
    """The reply's text pieces from Ollama, as they arrive.

    Closing this generator closes the upstream request, which stops generation.
    The socket teardown is shielded so a cancellation cannot interrupt it.
    """
    timeout = httpx.Timeout(CHAT_READ_TIMEOUT, connect=5)
    async with contextlib.AsyncExitStack() as stack:
        try:
            client = await stack.enter_async_context(httpx.AsyncClient(timeout=timeout))
            r = await stack.enter_async_context(client.stream(
                "POST", f"{OLLAMA}/api/chat", json={"model": MODEL, "messages": messages, "stream": True}
            ))
            if getattr(r, "status_code", 200) >= 400:
                await r.aread()  # so the error body can be read for the message
                r.raise_for_status()
            async for line in r.aiter_lines():
                piece = piece_from_line(line)
                if piece:
                    yield piece
        finally:
            with anyio.CancelScope(shield=True):
                await stack.aclose()


def piece_from_line(line: str) -> str:
    """The text in one line of Ollama's stream ('' for anything else); raises on an error line."""
    if not line:
        return ""
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return ""
    if not isinstance(data, dict):
        return ""
    if isinstance(data.get("error"), str):
        raise OllamaStreamError(data["error"][:200])
    message = data.get("message")
    piece = message.get("content") if isinstance(message, dict) else None
    return piece if isinstance(piece, str) else ""


def drop_message(message_id: int) -> None:
    """Remove exactly one stored message (the user's, when no reply came of it)."""
    with db() as c:
        c.execute("DELETE FROM messages WHERE id = ?", (message_id,))


def save_reply(session_id: int, text: str) -> None:
    with db() as c:
        # the session may have been deleted while the reply streamed
        c.execute(
            "INSERT INTO messages (session_id, role, content, created_at) "
            "SELECT ?,?,?,? WHERE EXISTS (SELECT 1 FROM sessions WHERE id = ?)",
            (session_id, "assistant", text, time.time(), session_id),
        )


class Reply:
    """One chat turn's cleanup, owned here so it works whether or not the body ever streams.

    ``finish`` is idempotent and is reached from the body generator's ``finally``
    (normal end, Stop, cancellation), from the response's background task (runs
    even when the body was never iterated) and from every failure path of
    ``chat``. It closes the upstream request (stops Ollama generating), then keeps
    the reply text if there is any, else takes the unanswered question back out.
    """

    def __init__(self, session_id: int, message_id: int):
        self.session_id = session_id
        self.message_id = message_id
        self.pieces = None  # the ollama_pieces generator, once there is one
        self.collected: list[str] = []
        self.done = False

    async def finish(self) -> None:
        if self.done:
            return
        self.done = True
        if self.pieces is not None:
            with anyio.CancelScope(shield=True):
                await self.pieces.aclose()
        text = "".join(self.collected)
        if text:
            save_reply(self.session_id, text)
        else:
            drop_message(self.message_id)


async def first_piece(pieces, request: Request | None):
    """The reply's first piece (None if Ollama sent none).

    Prompt evaluation can be silent for minutes, and the server is not told when
    the browser gives up meanwhile, so the client is polled while waiting;
    ClientGone if it left. Ollama's own errors propagate as they are.
    """
    if request is None:
        return await anext(pieces, None)
    outcome: list = []

    async with anyio.create_task_group() as tg:
        async def fetch():
            try:
                outcome.append((await anext(pieces, None), None))
            except Exception as e:  # noqa: BLE001 - re-raised below, outside the task group
                outcome.append((None, e))
            tg.cancel_scope.cancel()

        async def watch():
            while not await request.is_disconnected():
                await anyio.sleep(DISCONNECT_POLL_SECONDS)
            tg.cancel_scope.cancel()

        tg.start_soon(fetch)
        tg.start_soon(watch)
    if not outcome:
        raise ClientGone
    piece, error = outcome[0]
    if error:
        raise error
    return piece


def interruption_note(error: Exception) -> str:
    """Visible marker for a reply that Ollama cut short (kept in the stored text too)."""
    why = (f"Ollama returned an error: {error}" if isinstance(error, OllamaStreamError)
           else describe_ollama_error(error, MODEL, "chat"))
    return f"\n\n[Reply interrupted: {why}]"


@router.post("/api/chat")
async def chat(body: ChatIn, request: Request = None):
    # checked before anything is stored; the API can be called without the UI's trimming
    if not body.content.strip():
        raise HTTPException(400, "Message is empty")
    if len(body.content) > config.MAX_MESSAGE_CHARS:
        raise HTTPException(
            413, f"Message too long ({len(body.content)} characters, max {config.MAX_MESSAGE_CHARS})"
        )
    with db() as c:
        s = c.execute(
            "SELECT id, project_id FROM sessions WHERE id = ?", (body.session_id,)
        ).fetchone()
        if not s:
            raise HTTPException(404, "no such session")
        history = [
            {"role": r["role"], "content": r["content"]}
            for r in c.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
                (body.session_id,),
            )
        ]
        message_id = c.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
            (body.session_id, "user", body.content, time.time()),
        ).lastrowid
        # name the session after its first message, unless it was renamed already
        if not history:
            c.execute(
                "UPDATE sessions SET title = ? WHERE id = ? AND title = ?",
                (title_from_message(body.content), body.session_id, DEFAULT_TITLE),
            )

    # From here until the first reply text arrives, any failure must take the
    # stored question back out (reply.finish does that when nothing was collected):
    # an unanswered message would reappear on reload and be sent again as history.
    reply = Reply(body.session_id, message_id)
    try:
        # Project knowledge comes from retrieval, not from the session's window.
        status, hits = await retrieve(s["project_id"], build_retrieval_query(history, body.content))
        messages = []
        if hits:
            messages.append({"role": "system", "content": rag.format_hits(hits)})
        elif status in RAG_HINTS:
            messages.append({"role": "system", "content": RAG_HINTS[status]})
        everything = [*history, {"role": "user", "content": body.content}]
        window = fit_history(everything)
        messages.extend(window)

        reply.pieces = ollama_pieces(messages)
        try:
            first = await first_piece(reply.pieces, request)
        except (httpx.HTTPError, OllamaStreamError) as e:
            detail = (f"Ollama returned an error: {e}" if isinstance(e, OllamaStreamError)
                      else describe_ollama_error(e, MODEL, "chat"))
            raise HTTPException(502, detail) from e
        if first is None:
            raise HTTPException(502, "The model returned an empty reply")
    except ClientGone:
        await reply.finish()
        return Response(status_code=204)  # nobody is listening
    except BaseException:
        with anyio.CancelScope(shield=True):
            await reply.finish()
        raise
    reply.collected.append(first)

    async def stream():
        try:
            yield first
            try:
                async for piece in reply.pieces:
                    reply.collected.append(piece)
                    yield piece
            except (httpx.HTTPError, OllamaStreamError) as e:
                # Ollama dropped mid-reply: say so instead of passing off the text as complete
                note = interruption_note(e)
                reply.collected.append(note)
                yield note
        finally:
            # also runs when the client disconnects (Stop)
            await reply.finish()

    # json.dumps escapes non-ASCII, so the value is a valid header whatever the file names
    headers = {SOURCES_HEADER: json.dumps(list(dict.fromkeys(h["path"] for h in hits)))}
    if status:
        headers[STATUS_HEADER] = status
    if len(everything) > len(window):
        headers[TRIMMED_HEADER] = str(len(everything) - len(window))
    return StreamingResponse(
        stream(), media_type="text/plain", headers=headers, background=BackgroundTask(reply.finish)
    )
