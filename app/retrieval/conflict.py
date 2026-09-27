from __future__ import annotations

import re
from collections import defaultdict

from app.memory.models import MemoryRecord

_CURRENT_RE = re.compile(
    r"\b(now|current|currently|today|latest|present)\b|现在|目前|当前|最新",
    re.IGNORECASE,
)


def apply_current_state_bias(
    scores: dict[str, float],
    records_by_id: dict[str, MemoryRecord],
    query: str,
) -> dict[str, float]:
    """Demote older records that share the same subject/predicate.

    This is retrieval-time conflict resolution: history is preserved, but
    "current" questions prefer the newest version.
    """

    if not _CURRENT_RE.search(query or ""):
        return scores

    groups: dict[tuple[str, str], list[MemoryRecord]] = defaultdict(list)
    for record in records_by_id.values():
        if (
            record.subject
            and record.predicate
            and record.memory_type in {"fact", "profile", "preference", "event"}
        ):
            groups[(record.subject, record.predicate)].append(record)

    for group in groups.values():
        latest = max((record.timestamp or 0) for record in group)
        if latest <= 0:
            continue
        for record in group:
            if record.timestamp is not None and record.timestamp < latest:
                scores[record.id] = scores.get(record.id, 0.0) * 0.6

    return scores
