from __future__ import annotations

from app.memory.models import MemoryRecord
from app.retrieval.graph_jepa import build_predicted_adjacency


def make_record(record_id: str, embedding: list[float], entities: list[str], timestamp: int) -> MemoryRecord:
    return MemoryRecord(
        id=record_id,
        user_id="u1",
        session_id="s1",
        content=record_id,
        entities=entities,
        embedding=embedding,
        timestamp=timestamp,
    )


def test_graph_jepa_predicts_edges_between_similar_memories() -> None:
    records = [
        make_record("a", [1.0, 0.0], ["Shanghai", "user"], 1000),
        make_record("b", [0.9, 0.1], ["Shanghai", "move"], 2000),
        make_record("c", [0.0, 1.0], ["Paris"], 3000),
    ]
    adjacency = build_predicted_adjacency(
        records,
        query_vector=[1.0, 0.0],
        top_k=2,
        threshold=0.3,
        weight=1.0,
        max_nodes=10,
    )
    assert "a" in adjacency
    assert any(target == "b" for target, _ in adjacency["a"])
    assert "c" not in adjacency
