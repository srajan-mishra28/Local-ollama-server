"""Paths and tunable constants."""

from pathlib import Path

import rag

BASE = Path(__file__).resolve().parent.parent
DB_PATH = BASE / "server.db"
UPLOADS = BASE / "uploads"
UPLOADS.mkdir(exist_ok=True)

OLLAMA = "http://localhost:11434"
MODEL = "qwen2.5-coder:7b"

# Which files are worth pulling into context (shared with the RAG scan filter).
CODE_EXT, SKIP_DIRS = rag.CODE_EXT, rag.SKIP_DIRS
MAX_CONTEXT_BYTES = 120_000  # keep the prompt inside the model's window

# RAG: per-project index of file chunks, retrieved per question at chat time.
EMBED_MODEL = "nomic-embed-text"  # `ollama pull nomic-embed-text`
RAG_TOP_K = 5
RAG_CHUNK_CHARS = 1500
RAG_OVERLAP_CHARS = 200
RAG_SCAN_LIMIT = 10_000
EMBED_TIMEOUT = 120  # indexing: a cold model load can be slow
QUERY_EMBED_TIMEOUT = 10  # chat: don't stall the user waiting on retrieval
# Images are described by a vision model and the description is indexed.
# `ollama pull qwen2.5vl:7b` (alternatives: llava, minicpm-v). "" disables images.
VISION_MODEL = "qwen2.5vl:7b"
VISION_TIMEOUT = 300  # indexing: a vision model on CPU can take minutes per image
# Per-type size caps for indexed files (text files: rag.MAX_FILE_BYTES, 1 MB).
PDF_MAX_BYTES = 20_000_000
IMAGE_MAX_BYTES = 10_000_000
PDF_MAX_PAGES = 300  # pages beyond this are not indexed
PROJECT_FILES = BASE / "project_files"  # local copy of each project's indexed files
