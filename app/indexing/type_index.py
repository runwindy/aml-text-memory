from __future__ import annotations

import re

from app.memory.models import MemoryRecord

_PREFERENCE_RE = re.compile(
    r"\b(prefer|favorite|like|love|dislike|hate|preference)\b|喜欢|偏好|讨厌|不喜欢",
    re.IGNORECASE,
)
_TEMPORAL_RE = re.compile(
    r"\b(when|before|after|order|first|then|last|date|time)\b|时间|日期|先后|之前|之后|顺序",
    re.IGNORECASE,
)
_RULE_RE = re.compile(
    r"\b(rule|process|workflow|must|should|instruction|format)\b|规则|流程|必须|应该|格式",
    re.IGNORECASE,
)


def type_scores(records: list[MemoryRecord], query: str) -> dict[str, float]:
    weights = {
        "raw": 0.7,
        "fact": 1.0,
        "event": 0.9,
        "preference": 0.9,
        "profile": 0.9,
        "rule": 0.9,
        "summary": 0.8,
        "relation": 1.0,
    }

    if _PREFERENCE_RE.search(query or ""):
        weights.update({"preference": 1.4, "profile": 1.2, "fact": 0.8})
    if _TEMPORAL_RE.search(query or ""):
        weights.update({"event": 1.4, "fact": 0.8, "raw": 0.9})
    if _RULE_RE.search(query or ""):
        weights.update({"rule": 1.5, "summary": 1.1, "profile": 0.9})

    return {record.id: weights.get(record.memory_type, 0.7) for record in records}
