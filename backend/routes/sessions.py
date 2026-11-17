"""Chat sessions: list, create, delete, load, and per-session context/project."""

import json
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.database import db

router = APIRouter()


class NewSession(BaseModel):
    title: str = "New chat"
    project_id: int | None = None


@router.get("/api/sessions")
def list_sessions():
    with db() as c:
        rows = c.execute(
            "SELECT s.*, p.name AS project_name "
            "FROM sessions s LEFT JOIN projects p ON p.id = s.project_id "
            "ORDER BY s.created_at DESC, s.id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


@router.post("/api/sessions")
def create_session(body: NewSession):
    with db() as c:
        cur = c.execute(
            "INSERT INTO sessions (title, project_id, created_at) VALUES (?,?,?)",
            (body.title, body.project_id, time.time()),
        )
        return {"id": cur.lastrowid}


@router.delete("/api/sessions/{sid}")
def delete_session(sid: int):
    with db() as c:
        c.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
        c.execute("DELETE FROM sessions WHERE id = ?", (sid,))
    return {"ok": True}


@router.get("/api/sessions/{sid}")
def get_session(sid: int):
    with db() as c:
        s = c.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()
        if not s:
            raise HTTPException(404, "no such session")
        msgs = c.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
            (sid,),
        ).fetchall()
    out = dict(s)
    out["context"] = json.loads(out["context"])
    out["messages"] = [dict(m) for m in msgs]
    return out


class ContextUpdate(BaseModel):
    files: list[str]


@router.put("/api/sessions/{sid}/context")
def set_context(sid: int, body: ContextUpdate):
    with db() as c:
        c.execute(
            "UPDATE sessions SET context = ? WHERE id = ?",
            (json.dumps(body.files), sid),
        )
    return {"ok": True, "files": body.files}


# small endpoint the UI uses to bind a project to a session
@router.put("/api/sessions/{sid}/project")
def set_project(sid: int, pid: str = ""):
    val = int(pid) if pid.strip() else None
    with db() as c:
        c.execute("UPDATE sessions SET project_id = ? WHERE id = ?", (val, sid))
    return {"ok": True}
