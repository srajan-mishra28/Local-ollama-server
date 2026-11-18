"""
Local LLM server: entry point.

Features
  - Persistent chat sessions (SQLite)
  - Normal chats, plus projects: workspaces with their own sessions and files
  - RAG (rag.py): every file uploaded to a project is indexed (local copy,
    chunks, Ollama embeddings) and shared by all of the project's sessions
  - File transfer: phone -> laptop (and back)

Layout
  backend/   config, database, RAG store, prompt helpers and the API routers
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
from backend.routes import chat, files, health, project_files, projects, sessions
from backend.routes.chat import retrieve_context  # noqa: F401  (re-exported for tests)

STATIC = Path(__file__).resolve().parent / "static"

init_db()
app = FastAPI(title="Local LLM server")
for module in (sessions, projects, project_files, files, chat, health):
    app.include_router(module.router)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", media_type="text/html")
