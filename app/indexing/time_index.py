from __future__ import annotations

import re

from app.memory.models import MemoryRecord

_CURRENT_RE = re.compile(
    r"\b(now|current|currently|today|latest|present)\b|现在|目前|当前|最新",
    re.IGNORECASE,
)


class TimeIndex:
    """Simple recency/time intent index."""

    def __init__(self, records: list[MemoryRecord]) -> None:
        self.records = [record for record in records if record.timestamp is not None]
        self.min_ts = min((record.timestamp for record in self.records), default=None)
        self.max_ts = max((record.timestamp for record in self.records), default=None)

    def scores(self, query: str) -> dict[str, float]:
        if self.min_ts is None or self.max_ts is None:
            return {}
        span = max(self.max_ts - self.min_ts, 1)
        current = bool(_CURRENT_RE.search(query or ""))
        result: dict[str, float] = {}
        for record in self.records:
            recency = (record.timestamp - self.min_ts) / span
            result[record.id] = (0.5 + 0.5 * recency) if not current else recency
        return result
