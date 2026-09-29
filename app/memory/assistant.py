from __future__ import annotations

import re
from typing import Any, Sequence

from app.ingestion.models import CanonicalMessage

# Rule-based assistant-role signals.  These are deliberately lightweight:
# they do not replace the LLM extraction step, but they give the memory layer
# role-aware metadata and help the predictive layer decide whether an assistant
# reply is a confirmation, a summary, an update, or a novel extension.
_SIGNAL_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "confirmation": (
        re.compile(r"\b(yes|correct|right|exactly|understood|noted|got it|sure)\b", re.IGNORECASE),
        re.compile(r"(明白了|对的|是的|没错|收到|好的)"),
    ),
    "summary": (
        re.compile(r"\b(in summary|overall|to summarize|you mentioned|previously|earlier)\b", re.IGNORECASE),
        re.compile(r"(总结|总之|之前提到|前面提到|综合来看)"),
    ),
    "update": (
        re.compile(r"\b(now|currently|no longer|instead|changed|moved|updated|from now on)\b", re.IGNORECASE),
        re.compile(r"(现在|目前|不再|改成|搬到|更新|从此)"),
    ),
    "correction": (
        re.compile(r"\b(actually|i mean|correction|not exactly|wrong)\b", re.IGNORECASE),
        re.compile(r"(其实|纠正|不对|不是这样|我刚才说错)"),
    ),
    "elaboration": (
        re.compile(r"\b(because|for example|meaning|which means|in other words|specifically)\b", re.IGNORECASE),
        re.compile(r"(因为|例如|也就是说|具体来说|换句话说)"),
    ),
    "question": (
        re.compile(r"\?$"),
        re.compile(r"[？?]"),
    ),
}


def detect_assistant_signals(text: str) -> list[str]:
    """Return coarse assistant-role labels for one assistant message."""

    value = text or ""
    signals: list[str] = []
    for label, patterns in _SIGNAL_PATTERNS.items():
        if any(pattern.search(value) for pattern in patterns):
            signals.append(label)
    return signals


def assistant_metadata(messages: Sequence[CanonicalMessage]) -> dict[str, Any]:
    """Build role-aware metadata for a dialogue window.

    The predictive layer uses user_context_text as the context and
    assistant_target_text as the observed future target.  The raw text is kept
    so the signal can be recomputed or audited later.
    """

    user_messages = [message for message in messages if message.role == "user"]
    assistant_messages = [message for message in messages if message.role == "assistant"]

    assistant_signals: list[str] = []
    for message in assistant_messages:
        for signal in detect_assistant_signals(message.raw_content):
            if signal not in assistant_signals:
                assistant_signals.append(signal)

    roles: list[str] = []
    for message in messages:
        if message.role not in roles:
            roles.append(message.role)

    return {
        "source_roles": roles,
        "has_assistant": bool(assistant_messages),
        "assistant_message_ids": [message.message_id for message in assistant_messages],
        "assistant_signals": assistant_signals,
        "user_context_text": "\n".join(message.raw_content for message in user_messages),
        "assistant_target_text": "\n".join(message.raw_content for message in assistant_messages),
    }
