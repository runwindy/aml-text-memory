from __future__ import annotations

from collections import defaultdict, deque

from app.memory.models import MemoryRecord
from app.retrieval.query_keywords import QueryKeywords


def expand_graph(
    records: list[MemoryRecord],
    seed_ids: list[str],
    keywords: QueryKeywords,
    *,
    max_hops: int = 2,
) -> dict[str, float]:
    """Expand candidate memories through relation records.

    Relation records use:
      subject      = source memory id
      predicate    = relation type
      object_value = target memory id
    """

    by_id = {record.id: record for record in records}
    adjacency: dict[str, list[tuple[str, MemoryRecord, float]]] = defaultdict(list)

    for record in records:
        if record.memory_type != "relation":
            continue
        source = record.subject
        target = record.object_value
        if not source or not target:
            continue
        edge_weight = 1.0
        if keywords.relation_hints and record.predicate in keywords.relation_hints:
            edge_weight = 1.2
        adjacency[source].append((target, record, edge_weight))
        adjacency[target].append((source, record, edge_weight * 0.8))

    scores: dict[str, float] = {}
    visited: set[str] = set()
    queue: deque[tuple[str, int, float]] = deque(
        (seed_id, 0, 1.0) for seed_id in seed_ids
    )

    while queue:
        current_id, depth, weight = queue.popleft()
        if current_id in visited or depth >= max_hops:
            continue
        visited.add(current_id)

        for neighbor_id, relation_record, edge_weight in adjacency.get(current_id, []):
            if neighbor_id not in by_id:
                continue
            neighbor_score = weight * edge_weight
            scores[neighbor_id] = max(scores.get(neighbor_id, 0.0), neighbor_score)
            scores[relation_record.id] = max(
                scores.get(relation_record.id, 0.0),
                neighbor_score * 0.9,
            )
            if depth + 1 < max_hops:
                queue.append((neighbor_id, depth + 1, neighbor_score * 0.8))

    return scores
