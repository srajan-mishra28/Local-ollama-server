"""
Local LLM server: entry point.

Features
  - Persistent chat sessions (SQLite)
  - RAG over project files (rag.py): local copies, chunks, Ollama embeddings
  - File transfer: phone -> laptop (and back)
  - Project context: register a code folder, attach files to a session

Layout
  backend/   config, database, RAG store and the API routers
  static/    the web UI (index.html, style.css, app.js)
  rag.py     vector store used for project RAG

Run:
    pip install -r requirements.txt
    python -m uvicorn app:app --host 0.0.0.0 --port 8000

Then open http://<lan-ip>:8000
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.database import init_db
from backend.routes import chat, files, projects, rag_index, sessions
from backend.routes.chat import retrieve_context  # noqa: F401  (re-exported for tests)
from backend.routes.projects import scan_project  # noqa: F401  (re-exported for tests)

STATIC = Path(__file__).resolve().parent / "static"

init_db()
app = FastAPI(title="Local LLM server")
for module in (sessions, projects, rag_index, files, chat):
    app.include_router(module.router)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", media_type="text/html")
