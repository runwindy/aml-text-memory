from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

from app.memory.models import MemoryRecord
from app.retrieval.hybrid import cosine_similarity

_DAY_MS = 24 * 60 * 60 * 1000


def _entity_set(record: MemoryRecord) -> set[str]:
    values = {str(value).strip().lower() for value in (record.entities or []) if str(value).strip()}
    if record.subject:
        values.add(record.subject.strip().lower())
    if record.object_value:
        values.add(record.object_value.strip().lower())
    return values


def _entity_jaccard(left: MemoryRecord, right: MemoryRecord) -> float:
    a = _entity_set(left)
    b = _entity_set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _temporal_adjacency(left: MemoryRecord, right: MemoryRecord) -> float:
    if left.timestamp is None or right.timestamp is None:
        return 0.0
    distance_days = abs(left.timestamp - right.timestamp) / _DAY_MS
    return 1.0 / (1.0 + distance_days)


def _select_nodes(
    records: Sequence[MemoryRecord],
    query_vector: Sequence[float] | None,
    max_nodes: int,
) -> list[MemoryRecord]:
    with_embedding = [record for record in records if record.embedding]
    if not with_embedding:
        return []
    if query_vector is not None and len(query_vector) > 0:
        ranked = sorted(
            with_embedding,
            key=lambda record: cosine_similarity(query_vector, record.embedding or []),
            reverse=True,
        )
    else:
        ranked = list(with_embedding)
    return ranked[: max(1, max_nodes)]


def build_predicted_adjacency(
    records: Sequence[MemoryRecord],
    *,
    query_vector: Sequence[float] | None = None,
    top_k: int = 8,
    threshold: float = 0.55,
    weight: float = 0.5,
    max_nodes: int = 200,
) -> dict[str, list[tuple[str, float]]]:
    """Build Graph-JEPA style predicted edges between candidate memories.

    The score mixes latent similarity, entity overlap, and temporal adjacency.
    It is deliberately local: only the strongest nodes are considered, so the
    search path remains O(max_nodes^2) instead of scanning the whole user DB.
    """

    nodes = _select_nodes(records, query_vector, max_nodes)
    adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
    if len(nodes) < 2:
        return {}

    for index, left in enumerate(nodes):
        scored: list[tuple[float, MemoryRecord]] = []
        for right in nodes[index + 1 :]:
            latent = cosine_similarity(left.embedding or [], right.embedding or [])
            if latent <= 0:
                continue
            score = (
                0.60 * latent
                + 0.25 * _entity_jaccard(left, right)
                + 0.15 * _temporal_adjacency(left, right)
            )
            if score >= threshold:
                scored.append((score, right))

        scored.sort(key=lambda item: item[0], reverse=True)
        for score, right in scored[: max(1, top_k)]:
            edge_weight = max(0.0, min(1.0, score)) * weight
            adjacency[left.id].append((right.id, edge_weight))
            adjacency[right.id].append((left.id, edge_weight))

    return dict(adjacency)
