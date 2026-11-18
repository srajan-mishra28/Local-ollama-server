"""Turn failures talking to Ollama into messages a user can act on.

Raw httpx text ("All connection attempts failed", mdn links, ...) never reaches
the UI: every failure maps to one of a few plain sentences.
"""

import httpx

from backend.config import OLLAMA

MODEL_ROLES = {"chat": "chat", "embed": "embedding", "vision": "vision"}


def _ollama_detail(response: httpx.Response | None) -> str:
    """Ollama's own {"error": "..."} text when it is short, else ''."""
    try:
        detail = response.json().get("error", "")
    except Exception:  # no response, not read yet, not JSON: nothing to add
        return ""
    return detail if isinstance(detail, str) and 0 < len(detail) <= 200 else ""


def describe_ollama_error(exc: BaseException, model: str, role: str = "chat") -> str:
    """A one-sentence, user-readable message for an httpx failure.

    ``model`` is the model the request was for and ``role`` is 'chat', 'embed' or
    'vision'; both are only used to word a 'model not installed' (404) answer.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 404:
            return (f"The {MODEL_ROLES.get(role, role)} model '{model}' isn't installed. "
                    f"Run: ollama pull {model}")
        detail = _ollama_detail(exc.response)
        return f"Ollama returned an error ({status}): {detail}" if detail else f"Ollama returned an error ({status})."
    if isinstance(exc, (httpx.ConnectTimeout, httpx.ConnectError)):
        return f"Can't reach Ollama at {OLLAMA}. Is it running?"
    if isinstance(exc, httpx.TimeoutException):
        return "Ollama took too long to answer"
    if isinstance(exc, httpx.TransportError):
        return f"Lost the connection to Ollama at {OLLAMA}. Is it still running?"
    return "Ollama request failed."
