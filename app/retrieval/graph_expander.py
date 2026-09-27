from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Iterable

from app.memory.models import MemoryRecord
from app.retrieval.query_keywords import QueryKeywords


def _normalize_entity(value: str | None) -> str:
    return str(value or "").strip().lower()


def _edge_weight(record: MemoryRecord, keywords: QueryKeywords) -> float:
    weight = max(0.2, min(1.0, record.confidence or 0.7))
    if keywords.relation_hints and record.predicate in keywords.relation_hints:
        weight = min(1.0, weight * 1.3)
    return weight


def _build_adjacency(
    records: list[MemoryRecord],
    keywords: QueryKeywords,
) -> dict[str, list[tuple[str, MemoryRecord | None, float]]]:
    adjacency: dict[str, list[tuple[str, MemoryRecord | None, float]]] = defaultdict(list)

    # 1. Explicit relation records.
    for record in records:
        if record.memory_type != "relation":
            continue
        source = record.subject
        target = record.object_value
        if not source or not target:
            continue
        weight = _edge_weight(record, keywords)
        adjacency[source].append((target, record, weight))
        adjacency[target].append((source, record, weight * 0.8))

    # 2. Entity co-occurrence edges. Keep high-degree entities bounded.
    inverted: dict[str, list[str]] = defaultdict(list)
    for record in records:
        seen_entities: set[str] = set()
        for entity in record.entities or []:
            normalized = _normalize_entity(entity)
            if normalized and normalized not in seen_entities:
                seen_entities.add(normalized)
                inverted[normalized].append(record.id)
        if record.subject:
            normalized = _normalize_entity(record.subject)
            if normalized and normalized not in seen_entities:
                inverted[normalized].append(record.id)
        if record.object_value:
            normalized = _normalize_entity(record.object_value)
            if normalized and normalized not in seen_entities:
                inverted[normalized].append(record.id)

    for entity, record_ids in inverted.items():
        unique_ids = list(dict.fromkeys(record_ids))
        if len(unique_ids) < 2 or len(unique_ids) > 20:
            continue
        for index, source_id in enumerate(unique_ids):
            for target_id in unique_ids[index + 1 :]:
                adjacency[source_id].append((target_id, None, 0.25))
                adjacency[target_id].append((source_id, None, 0.25))

    return adjacency


def expand_graph(
    records: list[MemoryRecord],
    seed_ids: Iterable[str],
    keywords: QueryKeywords,
    *,
    max_hops: int = 2,
    beam: int = 10,
    decay: float = 0.8,
) -> dict[str, float]:
    """Beam-search over relation records and entity co-occurrence edges."""

    by_id = {record.id: record for record in records}
    adjacency = _build_adjacency(records, keywords)

    scores: dict[str, float] = {}
    frontier: list[tuple[float, int, str, float]] = []
    for seed_id in seed_ids:
        if seed_id in by_id:
            heapq.heappush(frontier, (-1.0, 0, seed_id, 1.0))

    expanded: set[str] = set()
    while frontier:
        neg_score, depth, current_id, weight = heapq.heappop(frontier)
        if current_id in expanded or depth >= max_hops:
            continue
        expanded.add(current_id)

        neighbors = adjacency.get(current_id, [])
        scored_neighbors: list[tuple[float, str, MemoryRecord | None]] = []
        for neighbor_id, relation_record, edge_weight in neighbors:
            if neighbor_id not in by_id or neighbor_id in expanded:
                continue
            neighbor_score = weight * edge_weight * decay
            scored_neighbors.append((neighbor_score, neighbor_id, relation_record))

        scored_neighbors.sort(key=lambda item: item[0], reverse=True)
        for neighbor_score, neighbor_id, relation_record in scored_neighbors[:beam]:
            scores[neighbor_id] = max(scores.get(neighbor_id, 0.0), neighbor_score)
            if relation_record is not None:
                scores[relation_record.id] = max(
                    scores.get(relation_record.id, 0.0),
                    neighbor_score * 0.9,
                )
            if depth + 1 < max_hops:
                heapq.heappush(
                    frontier,
                    (-neighbor_score, depth + 1, neighbor_id, neighbor_score),
                )

    return scores
