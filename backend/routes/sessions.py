"""Chat sessions. A session with no project is a normal chat; otherwise it belongs to one project."""

import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.database import db

router = APIRouter()


class NewSession(BaseModel):
    title: str = "New chat"
    project_id: int | None = None


def insert_session(c, title: str, project_id: int | None) -> int:
    """Create a session; 404 if it names a project that does not exist."""
    if project_id is not None and not c.execute(
        "SELECT 1 FROM projects WHERE id = ?", (project_id,)
    ).fetchone():
        raise HTTPException(404, "no such project")
    cur = c.execute(
        "INSERT INTO sessions (title, project_id, created_at) VALUES (?,?,?)",
        (title[:200], project_id, time.time()),
    )
    return cur.lastrowid


@router.get("/api/sessions")
def list_sessions():
    """Normal chats only (no project); a project's sessions come with the project."""
    with db() as c:
        rows = c.execute(
            "SELECT id, title, project_id, created_at FROM sessions "
            "WHERE project_id IS NULL ORDER BY created_at DESC, id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


@router.post("/api/sessions")
def create_session(body: NewSession = NewSession()):
    """A normal chat, or a session inside a project when ``project_id`` is given."""
    with db() as c:
        return {"id": insert_session(c, body.title, body.project_id)}


@router.delete("/api/sessions/{sid}")
def delete_session(sid: int):
    with db() as c:
        c.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
        c.execute("DELETE FROM sessions WHERE id = ?", (sid,))
    return {"ok": True}


@router.get("/api/sessions/{sid}")
def get_session(sid: int):
    with db() as c:
        s = c.execute(
            "SELECT s.id, s.title, s.project_id, s.created_at, p.name AS project_name "
            "FROM sessions s LEFT JOIN projects p ON p.id = s.project_id WHERE s.id = ?",
            (sid,),
        ).fetchone()
        if not s:
            raise HTTPException(404, "no such session")
        msgs = c.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
            (sid,),
        ).fetchall()
    out = dict(s)
    out["messages"] = [dict(m) for m in msgs]
    return out
