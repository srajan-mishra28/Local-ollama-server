"""Chat: prompt assembly (project RAG excerpts + windowed history) and the streamed Ollama reply."""

import json
import sqlite3
import time

import anyio
import httpx
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
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


async def retrieve(project_id: int | None, query: str) -> tuple[str | None, list[dict]]:
    """(status, hits): the excerpts for the prompt (best first, within the size budget).

    ``status`` is None for a normal chat, else 'used' (excerpts found), 'none'
    (indexed, but nothing scored >= RAG_MIN_SCORE), 'empty' (nothing indexed) or
    'failed'. Retrieval is best-effort: no failure here (embedder down, stale
    index, database trouble) may break plain chat, it is reported as 'failed'.
    """
    if not project_id:
        return None, []
    try:
        if not await run_in_threadpool(rag_store.store.has_index, project_id):
            return "empty", []
        hits = await run_in_threadpool(rag_store.store.query, project_id, query, RAG_TOP_K)
    except (sqlite3.Error, OSError, httpx.HTTPError, ValueError, KeyError, rag.ReindexRequired):
        return "failed", []
    relevant = [h for h in hits if h["score"] >= config.RAG_MIN_SCORE]
    hits = rag.fit_hits(relevant, RAG_MAX_EXCERPT_CHARS)
    return ("used" if hits else "none"), hits


async def retrieve_hits(project_id: int | None, query: str) -> list[dict]:
    return (await retrieve(project_id, query))[1]


async def retrieve_context(project_id: int | None, query: str) -> str:
    """``retrieve_hits`` rendered as a system-message block ('' if none)."""
    return rag.format_hits(await retrieve_hits(project_id, query))


async def ollama_pieces(messages: list[dict]):
    """The reply's text pieces from Ollama, as they arrive.

    Closing this generator closes the upstream request, which stops generation.
    """
    timeout = httpx.Timeout(CHAT_READ_TIMEOUT, connect=5)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST", f"{OLLAMA}/api/chat", json={"model": MODEL, "messages": messages, "stream": True}
        ) as r:
            if getattr(r, "status_code", 200) >= 400:
                await r.aread()  # so the error body can be read for the message
                r.raise_for_status()
            async for line in r.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict) and isinstance(data.get("error"), str):
                    raise OllamaStreamError(data["error"][:200])
                piece = data.get("message", {}).get("content", "")
                if piece:
                    yield piece


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


@router.post("/api/chat")
async def chat(body: ChatIn):
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
    # stored question back out: an unanswered message would reappear on reload
    # and be sent again as history.
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

        pieces = ollama_pieces(messages)
        try:
            first = await anext(pieces, None)
        except (httpx.HTTPError, OllamaStreamError) as e:
            detail = (f"Ollama returned an error: {e}" if isinstance(e, OllamaStreamError)
                      else describe_ollama_error(e, MODEL, "chat"))
            raise HTTPException(502, detail) from e
        if first is None:
            raise HTTPException(502, "The model returned an empty reply")
    except BaseException:
        drop_message(message_id)
        raise

    async def stream():
        collected = [first]
        try:
            yield first
            async for piece in pieces:
                collected.append(piece)
                yield piece
        except (httpx.HTTPError, OllamaStreamError):
            pass  # Ollama dropped mid-reply: keep what arrived
        finally:
            # also runs when the client disconnects (Stop): close the upstream
            # request so generation stops, then keep the partial reply
            with anyio.CancelScope(shield=True):
                await pieces.aclose()
            save_reply(body.session_id, "".join(collected))

    # json.dumps escapes non-ASCII, so the value is a valid header whatever the file names
    headers = {SOURCES_HEADER: json.dumps(list(dict.fromkeys(h["path"] for h in hits)))}
    if status:
        headers[STATUS_HEADER] = status
    if len(everything) > len(window):
        headers[TRIMMED_HEADER] = str(len(everything) - len(window))
    return StreamingResponse(stream(), media_type="text/plain", headers=headers)
