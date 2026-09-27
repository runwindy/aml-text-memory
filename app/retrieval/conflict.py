from __future__ import annotations

from collections import defaultdict

from app.memory.models import MemoryRecord
from app.retrieval.temporal_resolver import parse_temporal_query


def _is_historical(record: MemoryRecord) -> bool:
    return record.status == "superseded" or record.valid_to is not None


def _is_active(record: MemoryRecord) -> bool:
    return record.status == "active" and record.valid_to is None


def apply_current_state_bias(
    scores: dict[str, float],
    records_by_id: dict[str, MemoryRecord],
    query: str,
) -> dict[str, float]:
    """Version-aware conflict policy.

    - "now/current" questions prefer active/latest versions.
    - "before/past/used to" questions prefer historical versions.
    - Other questions keep history and rely on the fusion ranker.
    """

    temporal = parse_temporal_query(query)
    if not temporal.current and not temporal.past:
        return scores

    adjusted = dict(scores)

    if temporal.current:
        for record in records_by_id.values():
            if _is_historical(record):
                adjusted[record.id] = adjusted.get(record.id, 0.0) * 0.45
            elif _is_active(record):
                adjusted[record.id] = adjusted.get(record.id, 0.0) * 1.15

        groups: dict[tuple[str, str], list[MemoryRecord]] = defaultdict(list)
        for record in records_by_id.values():
            if record.subject and record.predicate and record.memory_type in {
                "fact",
                "profile",
                "preference",
                "event",
            }:
                groups[(record.subject, record.predicate)].append(record)

        for group in groups.values():
            latest = max((record.timestamp or 0) for record in group)
            if latest <= 0:
                continue
            for record in group:
                if record.timestamp is not None and record.timestamp < latest:
                    adjusted[record.id] = adjusted.get(record.id, 0.0) * 0.6

    if temporal.past:
        for record in records_by_id.values():
            if _is_active(record):
                adjusted[record.id] = adjusted.get(record.id, 0.0) * 0.55
            elif _is_historical(record):
                adjusted[record.id] = adjusted.get(record.id, 0.0) * 1.15

    return adjusted
