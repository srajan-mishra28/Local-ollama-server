"""The shared RAG store instance (see rag.py)."""

import rag
from backend.config import (
    BASE,
    EMBED_MODEL,
    EMBED_TIMEOUT,
    IMAGE_MAX_BYTES,
    OLLAMA,
    PDF_MAX_BYTES,
    PDF_MAX_PAGES,
    PROJECT_FILES,
    QUERY_EMBED_TIMEOUT,
    RAG_CHUNK_CHARS,
    RAG_OVERLAP_CHARS,
    VISION_MODEL,
    VISION_TIMEOUT,
)

store = rag.RagStore(
    BASE / "rag.db",
    PROJECT_FILES,
    rag.OllamaEmbedder(OLLAMA, EMBED_MODEL, EMBED_TIMEOUT),
    model=EMBED_MODEL,
    chunk_size=RAG_CHUNK_CHARS,
    overlap=RAG_OVERLAP_CHARS,
    query_embedder=rag.OllamaEmbedder(OLLAMA, EMBED_MODEL, QUERY_EMBED_TIMEOUT),
    captioner=rag.OllamaCaptioner(OLLAMA, VISION_MODEL, VISION_TIMEOUT) if VISION_MODEL else None,
    vision_model=VISION_MODEL,
    max_pdf_bytes=PDF_MAX_BYTES,
    max_image_bytes=IMAGE_MAX_BYTES,
    max_pdf_pages=PDF_MAX_PAGES,
)
