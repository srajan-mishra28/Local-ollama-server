"""Chat: prompt assembly (project RAG excerpts + windowed history) and the streamed Ollama reply."""

import json
import sqlite3
import time

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import rag
from backend import rag_store
from backend.config import MODEL, OLLAMA, RAG_MAX_EXCERPT_CHARS, RAG_TOP_K
from backend.database import db
from backend.prompt import build_retrieval_query, fit_history

router = APIRouter()

SOURCES_HEADER = "X-RAG-Sources"  # JSON list of the file names used for this answer


class ChatIn(BaseModel):
    session_id: int
    content: str


async def retrieve_hits(project_id: int | None, query: str) -> list[dict]:
    """The excerpts that go into the prompt (best first, within the size budget).

    Empty for a normal chat, a project with no index, or any retrieval failure:
    retrieval is best-effort, and no failure here (embedder down, stale index,
    database trouble) may break plain chat.
    """
    if not project_id:
        return []
    try:
        if not await run_in_threadpool(rag_store.store.has_index, project_id):
            return []
        hits = await run_in_threadpool(rag_store.store.query, project_id, query, RAG_TOP_K)
    except (sqlite3.Error, OSError, httpx.HTTPError, ValueError, KeyError, rag.ReindexRequired):
        return []
    return rag.fit_hits(hits, RAG_MAX_EXCERPT_CHARS)


async def retrieve_context(project_id: int | None, query: str) -> str:
    """``retrieve_hits`` rendered as a system-message block ('' if none)."""
    return rag.format_hits(await retrieve_hits(project_id, query))


@router.post("/api/chat")
async def chat(body: ChatIn):
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
        c.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
            (body.session_id, "user", body.content, time.time()),
        )
        # name the session after its first message
        if not history:
            c.execute(
                "UPDATE sessions SET title = ? WHERE id = ?",
                (body.content[:60], body.session_id),
            )

    messages = []
    # Project knowledge comes from retrieval, not from the session's window.
    hits = await retrieve_hits(s["project_id"], build_retrieval_query(history, body.content))
    if hits:
        messages.append({"role": "system", "content": rag.format_hits(hits)})
    messages.extend(fit_history([*history, {"role": "user", "content": body.content}]))
    sources = list(dict.fromkeys(h["path"] for h in hits))

    async def stream():
        collected = []
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA}/api/chat",
                    json={"model": MODEL, "messages": messages, "stream": True},
                ) as r:
                    async for line in r.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        piece = data.get("message", {}).get("content", "")
                        if piece:
                            collected.append(piece)
                            yield piece
        finally:
            if collected:
                with db() as c:
                    # the session may have been deleted while the reply streamed
                    c.execute(
                        "INSERT INTO messages (session_id, role, content, created_at) "
                        "SELECT ?,?,?,? WHERE EXISTS (SELECT 1 FROM sessions WHERE id = ?)",
                        (body.session_id, "assistant", "".join(collected), time.time(),
                         body.session_id),
                    )

    # json.dumps escapes non-ASCII, so the value is a valid header whatever the file names
    return StreamingResponse(
        stream(), media_type="text/plain", headers={SOURCES_HEADER: json.dumps(sources)}
    )


@router.get("/api/health")
def health():
    return {"ok": True, "model": MODEL}
