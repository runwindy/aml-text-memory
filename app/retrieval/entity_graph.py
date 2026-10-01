from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from app.memory.models import MemoryRecord
from app.retrieval.hybrid import tokenize


def _node_terms(node: dict) -> set[str]:
    terms = set(tokenize(str(node.get("canonical_name") or "")))
    for alias in node.get("aliases") or []:
        terms.update(tokenize(str(alias)))
    return terms


def entity_graph_scores(
    *,
    records: Sequence[MemoryRecord],
    query_text: str,
    entity_nodes: Sequence[dict],
    entity_edges: Sequence[dict],
    memory_entity_links: Sequence[dict],
    max_hops: int = 1,
    seed_k: int = 10,
    decay: float = 0.7,
) -> dict[str, float]:
    """Expand query-matching entities through persisted entity edges and map
    them back to linked memory records."""

    if not records or not entity_nodes:
        return {}

    query_tokens = set(tokenize(query_text))
    if not query_tokens:
        return {}

    terms_by_entity: dict[str, set[str]] = {}
    for node in entity_nodes:
        entity_id = str(node.get("entity_id"))
        terms_by_entity[entity_id] = _node_terms(node)

    seed_entity_ids: list[str] = []
    for entity_id, terms in terms_by_entity.items():
        overlap = len(query_tokens & terms)
        if overlap:
            seed_entity_ids.append((overlap, entity_id))
    seed_entity_ids.sort(reverse=True)
    seeds = [entity_id for _, entity_id in seed_entity_ids[: max(1, seed_k)]]
    if not seeds:
        return {}

    adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for edge in entity_edges:
        source = str(edge.get("source_entity_id"))
        target = str(edge.get("target_entity_id"))
        weight = float(edge.get("confidence") or 0.7)
        adjacency[source].append((target, weight))
        adjacency[target].append((source, weight))

    entity_scores: dict[str, float] = {entity_id: 1.0 for entity_id in seeds}
    frontier = list(seeds)
    for _ in range(max(1, max_hops)):
        next_frontier: list[str] = []
        for current in frontier:
            current_score = entity_scores.get(current, 0.0)
            for neighbor, weight in adjacency.get(current, []):
                candidate = current_score * decay * weight
                if candidate > entity_scores.get(neighbor, 0.0):
                    entity_scores[neighbor] = candidate
                    next_frontier.append(neighbor)
        frontier = next_frontier
        if not frontier:
            break

    record_ids = {record.id for record in records}
    scores: dict[str, float] = {}
    for link in memory_entity_links:
        memory_id = str(link.get("memory_id"))
        entity_id = str(link.get("entity_id"))
        if memory_id not in record_ids:
            continue
        score = entity_scores.get(entity_id, 0.0)
        if score > scores.get(memory_id, 0.0):
            scores[memory_id] = score
    return scores
