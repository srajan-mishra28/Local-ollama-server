"""Registered project folders: CRUD, file listing and content search."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import rag
from backend import rag_store
from backend.config import CODE_EXT, SKIP_DIRS
from backend.database import db

router = APIRouter()


class NewProject(BaseModel):
    name: str
    path: str


@router.get("/api/projects")
def list_projects():
    with db() as c:
        return [dict(r) for r in c.execute("SELECT * FROM projects ORDER BY name")]


@router.post("/api/projects")
def create_project(body: NewProject):
    p = Path(body.path).expanduser()
    if not p.is_dir():
        raise HTTPException(400, f"not a directory: {p}")
    with db() as c:
        cur = c.execute(
            "INSERT INTO projects (name, path) VALUES (?,?)", (body.name, str(p))
        )
        return {"id": cur.lastrowid}


@router.delete("/api/projects/{pid}")
def delete_project(pid: int):
    with db() as c:
        c.execute("DELETE FROM projects WHERE id = ?", (pid,))
    rag_store.store.delete_project(pid)
    return {"ok": True}


def project_root(pid: int) -> Path:
    with db() as c:
        row = c.execute("SELECT path FROM projects WHERE id = ?", (pid,)).fetchone()
    if not row:
        raise HTTPException(404, "no such project")
    return Path(row["path"])


def scan_project(root: Path, q: str = "", limit: int = 2000, text_only: bool = False) -> list[str]:
    """Relative paths of indexable files under root, optionally filtered by substring.

    ``text_only`` leaves out PDFs and images (what can be pinned as context).
    """
    out = []
    for f in root.rglob("*"):
        rel = f.relative_to(root).as_posix()
        if not rag_store.store.scannable(rel) or not f.is_file():
            continue
        if text_only and rag.file_kind(rel) != "text":
            continue
        if q and q.lower() not in rel.lower():
            continue
        out.append(rel)
        if len(out) >= limit:
            break
    return sorted(out)


@router.get("/api/projects/{pid}/files")
def project_files(pid: int, q: str = ""):
    """Relative paths of text files in the project that can be pinned, optionally filtered."""
    return scan_project(project_root(pid), q, text_only=True)


@router.get("/api/projects/{pid}/grep")
def project_grep(pid: int, q: str):
    """Files whose contents mention q. Useful for finding context to attach."""
    if not q.strip():
        return []
    root = project_root(pid)
    hits = []
    needle = q.lower()
    for f in root.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in CODE_EXT:
            continue
        if any(part in SKIP_DIRS for part in f.parts):
            continue
        try:
            if needle in f.read_text(encoding="utf-8", errors="ignore").lower():
                hits.append(str(f.relative_to(root)).replace("\\", "/"))
        except OSError:
            continue
        if len(hits) >= 50:
            break
    return sorted(hits)
