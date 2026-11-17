"""Per-project RAG index: (re)index, status, single-file sync, upload, query."""

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

import rag
from backend import rag_store
from backend.config import EMBED_MODEL, RAG_SCAN_LIMIT, RAG_TOP_K, VISION_MODEL
from backend.routes.files import safe_name
from backend.routes.projects import project_root, scan_project

router = APIRouter()


def rag_call(fn, *args):
    """Run a store call: bad paths -> 400, stale index -> 409, Ollama failures -> 502."""
    models = f"{EMBED_MODEL} / {VISION_MODEL}" if VISION_MODEL else EMBED_MODEL
    try:
        return fn(*args)
    except rag.ReindexRequired as e:
        raise HTTPException(409, str(e))
    except rag.EmbeddingError as e:  # a ValueError, so must come before it
        raise HTTPException(502, f"embedding failed (is {EMBED_MODEL} pulled?): {e}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Ollama request failed (is it running, are {models} pulled?): {e}")


@router.post("/api/projects/{pid}/index")
def index_project(pid: int):
    """(Re)index the project folder. Incremental: unchanged files are skipped,
    files that vanished from disk are removed from the index."""
    root = project_root(pid)
    rels = scan_project(root, limit=RAG_SCAN_LIMIT)
    # a capped scan is incomplete, so absent files aren't known to be gone
    stats = rag_store.store.index_project(pid, root, rels, prune=len(rels) < RAG_SCAN_LIMIT)
    if "aborted" in stats:
        raise HTTPException(502, f"{stats['aborted']} (is Ollama running?); partial: {stats}")
    return stats


@router.get("/api/projects/{pid}/index")
def index_status(pid: int):
    project_root(pid)
    return rag_store.store.list_files(pid)


@router.post("/api/projects/{pid}/index/file")
def index_file(pid: int, path: str):
    """Re-sync one file from the project folder: re-index it, or drop it if deleted."""
    return {"result": rag_call(rag_store.store.sync_file, pid, project_root(pid), path)}


@router.delete("/api/projects/{pid}/index/file")
def unindex_file(pid: int, path: str):
    """Remove a file's chunks and local copy from the index."""
    project_root(pid)
    return {"deleted": rag_call(rag_store.store.delete_file, pid, path)}


@router.post("/api/projects/{pid}/upload")
async def upload_project_file(pid: int, file: UploadFile = File(...), path: str = Form("")):
    """Store a file into the project's local copy and index it.

    ``path`` is the project-relative name (defaults to the upload's basename).
    Re-uploading the same path replaces the old content and chunks.
    """
    project_root(pid)
    rel = path or safe_name(file.filename or "upload.txt")
    cap = rag_store.store.max_bytes(rel)
    data = await file.read(cap + 1)  # never buffer more than the cap
    if len(data) > cap:
        raise HTTPException(413, f"file too large (max {cap} bytes for this file type)")
    result = await run_in_threadpool(rag_call, rag_store.store.put_file, pid, rel, data, "upload")
    return {"path": rel, "result": result}


@router.get("/api/projects/{pid}/rag")
def rag_query(pid: int, q: str, k: int = RAG_TOP_K):
    """Top-k indexed chunks for a query (debugging / direct retrieval)."""
    project_root(pid)
    return rag_call(rag_store.store.query, pid, q, max(1, min(k, 50)))
