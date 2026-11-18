"""Shared test setup."""
import pytest

from backend import config
from backend.routes import health


@pytest.fixture(autouse=True)
def isolate_from_real_ollama(monkeypatch):
    """No test talks to a real Ollama, and none depends on the relevance cut-off.

    The fake embedder's scores bear no relation to nomic-embed-text's, so the
    threshold is off (every hit counts) unless a test sets RAG_MIN_SCORE itself.
    The health probe is stubbed to 'Ollama unreachable' and its cache cleared.
    """
    monkeypatch.setattr(config, "RAG_MIN_SCORE", -1.0)

    async def unreachable():
        return None

    monkeypatch.setattr(health, "installed_models", unreachable)
    monkeypatch.setitem(health._cache, "at", None)
