from __future__ import annotations

import re
from collections import defaultdict

from app.memory.models import MemoryRecord

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def _tok(text: str) -> str:
    return text.casefold().strip()


class EntityIndex:
    """In-memory entity index for one user's candidate records."""

    def __init__(self, records: list[MemoryRecord]) -> None:
        self.mapping: dict[str, set[str]] = defaultdict(set)
        for record in records:
            values = set(record.entities or [])
            values.update(_TOKEN_RE.findall(record.content))
            for value in values:
                token = _tok(value)
                if token:
                    self.mapping[token].add(record.id)

    def scores(self, query: str) -> dict[str, float]:
        result: dict[str, float] = defaultdict(float)
        for token in set(_TOKEN_RE.findall(query or "")):
            for record_id in self.mapping.get(_tok(token), set()):
                result[record_id] += 1.0
        return dict(result)
