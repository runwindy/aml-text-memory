from __future__ import annotations

from collections import defaultdict

from app.memory.models import MemoryRecord


def _sort_key(record: MemoryRecord):
    ordinal = record.metadata.get("window_ordinal")
    if isinstance(ordinal, int):
        return (0, ordinal)
    return (1, record.timestamp or 0)


def expand_result_window(
    records: list[MemoryRecord],
    fused_scores: dict[str, float],
    seed_ids: list[str],
    *,
    window: int = 1,
    seed_k: int = 20,
    decay: float = 0.7,
) -> dict[str, float]:
    """Add same-session neighboring windows around the best seeds."""

    if window <= 0:
        return fused_scores

    by_id = {record.id: record for record in records}
    by_session: dict[str, list[MemoryRecord]] = defaultdict(list)
    for record in records:
        by_session[record.session_id].append(record)
    for session_records in by_session.values():
        session_records.sort(key=_sort_key)

    result = dict(fused_scores)
    for seed_id in seed_ids[:seed_k]:
        seed = by_id.get(seed_id)
        if seed is None:
            continue
        seed_score = result.get(seed_id, 0.0)
        siblings = by_session.get(seed.session_id, [])
        position = next((index for index, item in enumerate(siblings) if item.id == seed_id), None)
        if position is None:
            continue
        for distance in range(1, window + 1):
            for neighbor_position in (position - distance, position + distance):
                if 0 <= neighbor_position < len(siblings):
                    neighbor = siblings[neighbor_position]
                    neighbor_score = seed_score * (decay ** distance)
                    result[neighbor.id] = max(result.get(neighbor.id, 0.0), neighbor_score)
    return result
