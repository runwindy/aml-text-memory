from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.memory.models import MemoryRecord

_CURRENT_RE = re.compile(
    r"\b(now|current|currently|today|latest|present)\b|现在|目前|当前|最新",
    re.IGNORECASE,
)
_PAST_RE = re.compile(
    r"\b(before|after|past|previous|history|used to|formerly)\b|以前|过去|之前|之后|曾经",
    re.IGNORECASE,
)
_ORDER_RE = re.compile(
    r"\b(first|then|before|after|order|earlier|later)\b|先|然后|之前|之后|顺序",
    re.IGNORECASE,
)
_ABSOLUTE_RE = re.compile(
    r"\b(?:19|20)\d{2}\b|"
    r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b|"
    r"\b\d{1,2}:\d{2}\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class TemporalQuery:
    current: bool = False
    past: bool = False
    order: bool = False
    absolute_terms: list[str] = field(default_factory=list)


def parse_temporal_query(query: str) -> TemporalQuery:
    return TemporalQuery(
        current=bool(_CURRENT_RE.search(query or "")),
        past=bool(_PAST_RE.search(query or "")),
        order=bool(_ORDER_RE.search(query or "")),
        absolute_terms=_ABSOLUTE_RE.findall(query or ""),
    )


def temporal_scores(records: list[MemoryRecord], query: str) -> dict[str, float]:
    temporal = parse_temporal_query(query)
    if not (temporal.current or temporal.past or temporal.order or temporal.absolute_terms):
        return {}

    timestamps = [record.timestamp for record in records if record.timestamp is not None]
    min_ts = min(timestamps) if timestamps else None
    max_ts = max(timestamps) if timestamps else None
    span = max((max_ts - min_ts), 1) if min_ts is not None and max_ts is not None else 1

    scores: dict[str, float] = {}
    for record in records:
        score = 0.0
        is_historical = record.status == "superseded" or record.valid_to is not None
        is_active = record.status == "active" and record.valid_to is None

        if temporal.current:
            score += 0.75 if is_active else 0.15
            if record.timestamp is not None and min_ts is not None and max_ts is not None:
                score += 0.25 * ((record.timestamp - min_ts) / span)

        if temporal.past:
            score += 0.75 if is_historical else 0.15

        if temporal.order and record.memory_type in {"event", "fact", "relation"}:
            score += 0.4

        for term in temporal.absolute_terms:
            haystacks = [
                record.content,
                record.valid_from or "",
                str(record.timestamp or ""),
                str(record.metadata.get("time_text") or ""),
            ]
            if any(term.lower() in haystack.lower() for haystack in haystacks):
                score += 0.8

        if score > 0:
            scores[record.id] = min(score, 1.0)

    return scores
