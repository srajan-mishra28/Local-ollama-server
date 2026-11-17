"""Pure helpers that shape what is sent to the model: retrieval query, history window."""

from backend.config import MAX_HISTORY_CHARS, RAG_QUERY_HISTORY_CHARS, RAG_QUERY_MAX_CHARS


def build_retrieval_query(
    history: list[dict],
    latest: str,
    history_chars: int = RAG_QUERY_HISTORY_CHARS,
    max_chars: int = RAG_QUERY_MAX_CHARS,
) -> str:
    """Text to embed for retrieval: recent user turns, then the latest message.

    A follow-up such as "and the second one?" means nothing on its own, so the
    newest earlier user messages that fit in ``history_chars`` are prepended
    (oldest first; the oldest one kept may be cut to its tail). Assistant
    replies are left out: they are long and would drown the question. The
    whole query is capped at ``max_chars``, favouring the latest message.
    """
    latest = latest.strip()[:max_chars]
    room = min(history_chars, max_chars - len(latest))
    earlier: list[str] = []
    for m in reversed(history):
        if room <= 0:
            break
        if m["role"] != "user" or not m["content"].strip():
            continue
        text = m["content"].strip()
        earlier.append(text[-room:])
        room -= len(text) + 1  # +1 for the newline between turns
    return "\n".join([*reversed(earlier), latest])


def fit_history(messages: list[dict], budget: int = MAX_HISTORY_CHARS) -> list[dict]:
    """The newest messages that fit ``budget`` characters, in original order.

    ``messages`` ends with the latest user message, which is always kept even
    if it alone exceeds the budget. Older messages are added newest first and
    the window stops at the first one that does not fit, so what is kept is
    always a contiguous recent stretch of the conversation.
    """
    if not messages:
        return []
    kept = [messages[-1]]
    used = len(messages[-1]["content"])
    for m in reversed(messages[:-1]):
        used += len(m["content"])
        if used > budget:
            break
        kept.append(m)
    return kept[::-1]
