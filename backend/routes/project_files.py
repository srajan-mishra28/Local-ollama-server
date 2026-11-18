"""A project's files: every upload is indexed for RAG; list, delete and debug-query."""

import re

import httpx
from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

import rag
from backend import rag_store
from backend.config import EMBED_MODEL, MAX_UPLOAD_FILES, RAG_TOP_K
from backend.ollama_errors import describe_ollama_error
from backend.routes.projects import require_project

router = APIRouter()

MAX_NAME_LENGTH = 200  # also bounds the X-RAG-Sources response header


def rag_call(fn, *args):
    """Run a store call: bad paths -> 400, stale index -> 409, Ollama failures -> 502."""
    try:
        return fn(*args)
    except rag.ReindexRequired as e:
        raise HTTPException(409, f"{e}. Upload the project's files again to re-index them.")
    except rag.EmbeddingError as e:  # a ValueError, so must come before it
        raise HTTPException(502, f"The embedding model '{EMBED_MODEL}' gave an unusable answer: {e}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except OSError as e:  # names valid here but illegal on this filesystem (`*`, `?`, too long)
        raise HTTPException(400, f"file name not usable on this server: {e}")
    except httpx.HTTPError as e:
        # a 404 here is always the embedder: a missing vision model is a skipped image, not an error
        raise HTTPException(502, describe_ollama_error(e, EMBED_MODEL, "embed"))


def upload_name(filename: str | None) -> str:
    """The file's base name, whichever separator the client's OS used."""
    return re.split(r"[\\/]", filename or "")[-1] or "upload.txt"


async def index_upload(pid: int, file: UploadFile) -> dict:
    """Store and index one uploaded file; the outcome is reported, never raised.

    ``result`` is the store's outcome ('indexed', 'unchanged', 'skipped...') or
    'error' with the HTTP-style ``status`` and a ``detail`` message.
    """
    name = upload_name(file.filename)
    try:
        if len(name) > MAX_NAME_LENGTH:
            raise HTTPException(400, f"name too long (max {MAX_NAME_LENGTH} characters)")
        cap = rag_store.store.max_bytes(name)
        data = await file.read(cap + 1)  # never buffer more than the cap
        if len(data) > cap:
            raise HTTPException(413, f"file too large (max {cap} bytes for this file type)")
        result = await run_in_threadpool(rag_call, rag_store.store.put_file, pid, name, data)
    except HTTPException as e:
        return {"path": name, "result": "error", "status": e.status_code, "detail": e.detail}
    return {"path": name, "result": result}


@router.post("/api/projects/{pid}/files")
async def upload_project_files(pid: int, files: list[UploadFile]):
    """Upload one or more files (multipart field ``files``); each is indexed at once.

    Re-uploading a name replaces the earlier content and chunks. The response
    is 200 with one entry per file, so one bad file never hides the others'
    outcomes.
    """
    require_project(pid)
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(400, f"too many files (max {MAX_UPLOAD_FILES} per upload)")
    return {"results": [await index_upload(pid, f) for f in files]}


@router.get("/api/projects/{pid}/files")
def list_project_files(pid: int):
    require_project(pid)
    return rag_store.store.list_files(pid)


@router.delete("/api/projects/{pid}/files")
def delete_project_file(pid: int, path: str):
    """Remove a file's chunks and local copy."""
    require_project(pid)
    return {"deleted": rag_call(rag_store.store.delete_file, pid, path)}


@router.get("/api/projects/{pid}/rag")
def rag_query(pid: int, q: str, k: int = RAG_TOP_K):
    """Top-k indexed chunks for a query (debugging / direct retrieval)."""
    require_project(pid)
    return rag_call(rag_store.store.query, pid, q, max(1, min(k, 50)))
