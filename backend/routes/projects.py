"""Projects: named workspaces that own sessions and uploaded files."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend import rag_store
from backend.config import PROJECT_NAME_MAX
from backend.database import db
from backend.routes.sessions import NewSession, insert_session

router = APIRouter()


class NewProject(BaseModel):
    name: str


def require_project(pid: int) -> dict:
    """The project row, or 404."""
    with db() as c:
        row = c.execute("SELECT id, name FROM projects WHERE id = ?", (pid,)).fetchone()
    if not row:
        raise HTTPException(404, "no such project")
    return dict(row)


@router.get("/api/projects")
def list_projects():
    with db() as c:
        rows = c.execute("SELECT id, name FROM projects ORDER BY name COLLATE NOCASE, id").fetchall()
    return [dict(r) for r in rows]


@router.post("/api/projects")
def create_project(body: NewProject):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "project name is required")
    if len(name) > PROJECT_NAME_MAX:
        raise HTTPException(400, f"project name is too long (max {PROJECT_NAME_MAX} characters)")
    with db() as c:
        taken = {r["name"].strip().casefold() for r in c.execute("SELECT name FROM projects")}
        if name.casefold() in taken:
            raise HTTPException(409, f"A project named '{name}' already exists")
        # path is a legacy column from when projects were disk folders
        cur = c.execute("INSERT INTO projects (name, path) VALUES (?, '')", (name,))
        return {"id": cur.lastrowid, "name": name}


@router.get("/api/projects/{pid}")
def get_project(pid: int):
    """The project with its sessions (newest first) and its indexed files."""
    project = require_project(pid)
    with db() as c:
        sessions = c.execute(
            "SELECT id, title, project_id, created_at FROM sessions WHERE project_id = ? "
            "ORDER BY created_at DESC, id DESC",
            (pid,),
        ).fetchall()
    project["sessions"] = [dict(s) for s in sessions]
    project["files"] = rag_store.store.list_files(pid)
    return project


@router.delete("/api/projects/{pid}")
def delete_project(pid: int):
    """Delete the project with its sessions, messages, files, chunks and copies."""
    with db() as c:
        c.execute(
            "DELETE FROM messages WHERE session_id IN (SELECT id FROM sessions WHERE project_id = ?)",
            (pid,),
        )
        c.execute("DELETE FROM sessions WHERE project_id = ?", (pid,))
        c.execute("DELETE FROM projects WHERE id = ?", (pid,))
    rag_store.store.delete_project(pid)
    return {"ok": True}


@router.post("/api/projects/{pid}/sessions")
def create_project_session(pid: int, body: NewSession | None = None):
    """A new session inside the project (any ``project_id`` in the body is ignored)."""
    require_project(pid)
    with db() as c:
        return {"id": insert_session(c, (body.title if body else "New chat"), pid)}
