"""GET /api/health: liveness of this server plus whether Ollama and its models are usable."""

import time

import httpx
from fastapi import APIRouter

from backend import config
from backend.config import EMBED_MODEL, MODEL, OLLAMA, VISION_MODEL

router = APIRouter()

_cache: dict = {"at": None, "data": None}


async def installed_models() -> set[str] | None:
    """Names of the models Ollama has installed, or None if it cannot be reached."""
    try:
        async with httpx.AsyncClient(timeout=config.HEALTH_TIMEOUT) as client:
            r = await client.get(f"{OLLAMA}/api/tags")
            r.raise_for_status()
            return {m["name"] for m in r.json().get("models", []) if isinstance(m.get("name"), str)}
    except (httpx.HTTPError, ValueError, AttributeError, TypeError):
        return None


def has_model(installed: set[str], wanted: str) -> bool:
    """Ollama lists 'name:tag'; a bare configured name means ':latest'."""
    return wanted in installed or (":" not in wanted and f"{wanted}:latest" in installed)


async def probe() -> dict:
    installed = await installed_models()
    reachable = installed is not None
    have = (lambda m: has_model(installed, m)) if reachable else (lambda m: False)
    models = {
        "chat": have(MODEL),
        "embed": have(EMBED_MODEL),
        "vision": have(VISION_MODEL) if VISION_MODEL else None,
    }
    # The vision model is optional (images only), so it does not degrade the status.
    degraded = not (reachable and models["chat"] and models["embed"])
    return {
        "ok": True,  # this process is up
        "status": "degraded" if degraded else "ok",
        "model": MODEL,
        "max_message_chars": config.MAX_MESSAGE_CHARS,
        "embed_model": EMBED_MODEL,
        "vision_model": VISION_MODEL,
        "ollama": reachable,
        "models": models,
    }


@router.get("/api/health")
async def health():
    now = time.monotonic()
    if _cache["at"] is None or now - _cache["at"] >= config.HEALTH_CACHE_SECONDS:
        _cache["data"], _cache["at"] = await probe(), now
    return _cache["data"]
