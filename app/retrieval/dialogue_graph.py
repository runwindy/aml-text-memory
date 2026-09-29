from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Sequence

from app.memory.models import MemoryRecord


def _record_message_ids(record: MemoryRecord) -> list[str]:
    values = list(record.source_message_ids or [])
    if not values:
        values = list(record.metadata.get("source_message_ids") or [])
    return [str(value) for value in values if value]


def build_tree_graph_scores(
    *,
    records: Sequence[MemoryRecord],
    dialogue_nodes: Sequence[dict],
    memory_edges: Sequence[dict],
    seed_scores: dict[str, float],
    decay: float = 0.7,
    max_hops: int = 2,
    max_expansions: int = 100,
) -> dict[str, float]:
    """Expand retrieval seeds through persisted dialogue tree and graph edges."""

    if not records or not seed_scores:
        return {}

    records_by_id = {record.id: record for record in records}
    record_ids_by_message_id: dict[str, set[str]] = defaultdict(set)
    for record in records:
        for message_id in _record_message_ids(record):
            record_ids_by_message_id[message_id].add(record.id)

    nodes_by_id = {str(node.get("node_id")): node for node in dialogue_nodes}
    nodes_by_message_id: dict[str, dict] = {}
    nodes_by_parent: dict[str, list[dict]] = defaultdict(list)
    for node in dialogue_nodes:
        metadata = node.get("metadata") or {}
        message_id = metadata.get("message_id")
        if message_id:
            nodes_by_message_id[str(message_id)] = node
        parent_id = node.get("parent_id")
        if parent_id:
            nodes_by_parent[str(parent_id)].append(node)

    edges_by_node: dict[str, list[dict]] = defaultdict(list)
    edges_by_record: dict[str, list[dict]] = defaultdict(list)
    for edge in memory_edges:
        source_id = str(edge.get("source_id", ""))
        target_id = str(edge.get("target_id", ""))
        if source_id in nodes_by_id or target_id in nodes_by_id:
            edges_by_node[source_id].append(edge)
            edges_by_node[target_id].append(edge)
        else:
            edges_by_record[source_id].append(edge)
            edges_by_record[target_id].append(edge)

    best: dict[str, float] = {}
    heap: list[tuple[float, int, str]] = []
    for record_id, score in seed_scores.items():
        if record_id in records_by_id:
            heapq.heappush(heap, (-float(score), 0, record_id))

    expansions = 0
    while heap and expansions < max_expansions:
        neg_score, depth, record_id = heapq.heappop(heap)
        score = -neg_score
        if score <= best.get(record_id, -1.0):
            continue
        best[record_id] = score
        if depth >= max_hops:
            continue

        record = records_by_id.get(record_id)
        if record is None:
            continue
        expansions += 1

        def add_record(target_record_id: str, bonus: float) -> None:
            if target_record_id not in records_by_id:
                return
            candidate_score = score * decay * bonus
            if candidate_score > best.get(target_record_id, -1.0):
                heapq.heappush(heap, (-candidate_score, depth + 1, target_record_id))

        # 1. Explicit memory-to-memory edges (for example supersedes).
        for edge in edges_by_record.get(record_id, []):
            source_id = str(edge.get("source_id"))
            target_id = str(edge.get("target_id"))
            other_id = target_id if source_id == record_id else source_id
            add_record(other_id, float(edge.get("confidence") or 0.7))

        # 2. Dialogue tree and node-level graph expansion.
        node_ids = [
            nodes_by_message_id[message_id].get("node_id")
            for message_id in _record_message_ids(record)
            if message_id in nodes_by_message_id
        ]
        for node_id in [str(value) for value in node_ids if value]:
            node = nodes_by_id.get(node_id)
            if node is None:
                continue

            related_nodes = []
            parent_id = node.get("parent_id")
            if parent_id:
                related_nodes.extend(nodes_by_parent.get(str(parent_id), []))
            prev_id = node.get("prev_id")
            next_id = node.get("next_id")
            if prev_id and str(prev_id) in nodes_by_id:
                related_nodes.append(nodes_by_id[str(prev_id)])
            if next_id and str(next_id) in nodes_by_id:
                related_nodes.append(nodes_by_id[str(next_id)])

            for edge in edges_by_node.get(node_id, []):
                source_id = str(edge.get("source_id"))
                target_id = str(edge.get("target_id"))
                other_id = target_id if source_id == node_id else source_id
                if other_id in nodes_by_id:
                    related_nodes.append(nodes_by_id[other_id])
                    related_weight = float(edge.get("confidence") or 0.7)
                else:
                    related_weight = 0.5
                # Record-level edge discovered through a node.
                if source_id in records_by_id:
                    add_record(source_id, related_weight)
                if target_id in records_by_id:
                    add_record(target_id, related_weight)

            for related_node in related_nodes:
                related_message_id = (related_node.get("metadata") or {}).get("message_id")
                if not related_message_id:
                    continue
                for related_record_id in record_ids_by_message_id.get(str(related_message_id), set()):
                    add_record(related_record_id, 0.9)

    return {
        record_id: score
        for record_id, score in best.items()
        if record_id not in seed_scores
    }
