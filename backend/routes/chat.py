"""Chat: prompt assembly (pinned files + RAG excerpts) and the streamed Ollama reply."""

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
from backend.config import MAX_CONTEXT_BYTES, MODEL, OLLAMA, RAG_TOP_K
from backend.database import db
from backend.routes.projects import project_root

router = APIRouter()


class ChatIn(BaseModel):
    session_id: int
    content: str


def build_context_block(project_id: int | None, files: list[str]) -> str:
    if not project_id or not files:
        return ""
    root = project_root(project_id)
    parts, total = [], 0
    for rel in files:
        if rag.file_kind(rel) != "text":
            continue  # never inject raw PDF/image bytes; those are reachable via RAG
        f = (root / rel).resolve()
        # refuse anything that escapes the project root
        if root.resolve() not in f.parents and f != root.resolve():
            continue
        if not f.is_file():
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if total + len(text) > MAX_CONTEXT_BYTES:
            text = text[: max(0, MAX_CONTEXT_BYTES - total)]
        total += len(text)
        parts.append(f"--- {rel} ---\n{text}")
        if total >= MAX_CONTEXT_BYTES:
            break
    if not parts:
        return ""
    return (
        "You are helping with the following codebase. Use these files as the "
        "source of truth; say so if something is not in them.\n\n"
        + "\n\n".join(parts)
    )


async def retrieve_context(project_id: int | None, question: str) -> str:
    """RAG excerpts for the question, or '' if there is no index or retrieval fails.

    Retrieval is best-effort: no failure here (embedder down, stale index,
    database trouble) may break plain chat.
    """
    if not project_id:
        return ""
    try:
        if not await run_in_threadpool(rag_store.store.has_index, project_id):
            return ""
        hits = await run_in_threadpool(rag_store.store.query, project_id, question, RAG_TOP_K)
    except (sqlite3.Error, OSError, httpx.HTTPError, ValueError, KeyError, rag.ReindexRequired):
        return ""
    return rag.format_hits(hits)


@router.post("/api/chat")
async def chat(body: ChatIn):
    with db() as c:
        s = c.execute("SELECT * FROM sessions WHERE id = ?", (body.session_id,)).fetchone()
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
    ctx = build_context_block(s["project_id"], json.loads(s["context"]))
    if ctx:
        messages.append({"role": "system", "content": ctx})
    hits = await retrieve_context(s["project_id"], body.content)
    if hits:
        messages.append({"role": "system", "content": hits})
    messages.extend(history)
    messages.append({"role": "user", "content": body.content})

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
                    c.execute(
                        "INSERT INTO messages (session_id, role, content, created_at) "
                        "VALUES (?,?,?,?)",
                        (body.session_id, "assistant", "".join(collected), time.time()),
                    )

    return StreamingResponse(stream(), media_type="text/plain")


@router.get("/api/health")
def health():
    return {"ok": True, "model": MODEL}
