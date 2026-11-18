"""Paths and tunable constants."""

from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DB_PATH = BASE / "server.db"
UPLOADS = BASE / "uploads"
UPLOADS.mkdir(exist_ok=True)

OLLAMA = "http://localhost:11434"
MODEL = "qwen2.5-coder:7b"

# Session history window: the newest messages that fit this many characters
# (plus always the latest message) go into the prompt. Older turns stay stored
# and visible in the UI but are not sent. Keep it inside the model's context.
MAX_HISTORY_CHARS = 24_000
# Longest message POST /api/chat accepts (413 beyond it). A huge message makes
# a local model stall for minutes and pushes all earlier history out of the prompt.
MAX_MESSAGE_CHARS = 20_000
CHAT_READ_TIMEOUT = 300  # seconds of silence from Ollama (prompt processing included) before giving up
HEALTH_TIMEOUT = 2  # /api/health asks Ollama for its model list with this timeout
HEALTH_CACHE_SECONDS = 5  # ...and reuses the answer this long, so polling is cheap

# RAG: per-project index of uploaded file chunks, shared by all the project's
# sessions and retrieved per question at chat time.
EMBED_MODEL = "nomic-embed-text"  # `ollama pull nomic-embed-text`
RAG_TOP_K = 5
# Minimum cosine similarity for a chunk to be used. Measured with nomic-embed-text
# (short questions against short paragraphs): on-topic chunks scored 0.59-0.79,
# off-topic ones 0.25-0.44, and a code question against unrelated code of the same
# language 0.52. 0.5 sits between; below it nothing is injected or listed.
# Longer chunks score lower, so re-measure (GET /api/projects/{id}/rag?q=...) if
# answers stop finding files, or if you change EMBED_MODEL.
RAG_MIN_SCORE = 0.5
RAG_MAX_EXCERPT_CHARS = 6_000  # total size of the excerpts injected per question
# The retrieval query is the latest message plus a little recent conversation,
# so follow-ups ("and the second one?") still find the right chunks.
RAG_QUERY_HISTORY_CHARS = 400  # budget for earlier user turns in the query
RAG_QUERY_MAX_CHARS = 2_000  # cap on the whole query text sent to the embedder
MAX_UPLOAD_FILES = 20  # files accepted by one project upload request
PROJECT_NAME_MAX = 100
RAG_CHUNK_CHARS = 1500
RAG_OVERLAP_CHARS = 200
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
