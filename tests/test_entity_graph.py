from __future__ import annotations

from app.memory.models import MemoryRecord
from app.retrieval.entity_graph import entity_graph_scores


def test_entity_graph_scores_map_query_to_linked_memory() -> None:
    records = [
        MemoryRecord(id="m1", user_id="u", session_id="s", content="Shanghai"),
        MemoryRecord(id="m2", user_id="u", session_id="s", content="Beijing"),
    ]
    nodes = [
        {"entity_id": "e1", "canonical_name": "shanghai", "aliases": ["上海"]},
        {"entity_id": "e2", "canonical_name": "beijing", "aliases": ["北京"]},
    ]
    edges = [
        {"source_entity_id": "e1", "target_entity_id": "e2", "confidence": 0.8},
    ]
    links = [
        {"memory_id": "m1", "entity_id": "e1"},
        {"memory_id": "m2", "entity_id": "e2"},
    ]
    scores = entity_graph_scores(
        records=records,
        query_text="Where is Shanghai?",
        entity_nodes=nodes,
        entity_edges=edges,
        memory_entity_links=links,
        max_hops=1,
    )
    assert scores.get("m1", 0) > 0
    assert scores.get("m2", 0) > 0
