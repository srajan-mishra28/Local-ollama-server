"""File transfer between devices and the host's uploads/ folder."""

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.config import UPLOADS

router = APIRouter()


def safe_name(name: str) -> str:
    return Path(name).name.replace("..", "_")


@router.post("/api/upload")
async def upload(file: UploadFile = File(...), note: str = Form("")):
    dest = UPLOADS / safe_name(file.filename or "upload.bin")
    stem, suffix = dest.stem, dest.suffix
    i = 1
    while dest.exists():
        dest = UPLOADS / f"{stem}_{i}{suffix}"
        i += 1
    with dest.open("wb") as out:
        while chunk := await file.read(1 << 20):
            out.write(chunk)
    return {"name": dest.name, "size": dest.stat().st_size, "note": note}


@router.get("/api/files")
def list_files():
    out = []
    for f in sorted(UPLOADS.iterdir(), key=lambda p: -p.stat().st_mtime):
        if f.is_file():
            out.append({"name": f.name, "size": f.stat().st_size, "mtime": f.stat().st_mtime})
    return out


@router.get("/api/files/{name}")
def download(name: str):
    f = UPLOADS / safe_name(name)
    if not f.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(f, filename=f.name)


@router.delete("/api/files/{name}")
def delete_file(name: str):
    f = UPLOADS / safe_name(name)
    if f.is_file():
        f.unlink()
    return {"ok": True}
