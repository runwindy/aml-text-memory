from __future__ import annotations

import sqlite3

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.memory.models import MemoryRecord
from app.retrieval.dialogue_graph import build_tree_graph_scores


def test_build_tree_graph_scores_expands_siblings() -> None:
    records = [
        MemoryRecord(id="a", user_id="u", session_id="s", content="a", source_message_ids=["m0"]),
        MemoryRecord(id="b", user_id="u", session_id="s", content="b", source_message_ids=["m1"]),
    ]
    nodes = [
        {"node_id": "turn", "parent_id": None, "prev_id": None, "next_id": None, "metadata": {}},
        {"node_id": "n0", "parent_id": "turn", "prev_id": None, "next_id": "n1", "metadata": {"message_id": "m0"}},
        {"node_id": "n1", "parent_id": "turn", "prev_id": "n0", "next_id": None, "metadata": {"message_id": "m1"}},
    ]
    scores = build_tree_graph_scores(
        records=records,
        dialogue_nodes=nodes,
        memory_edges=[],
        seed_scores={"a": 1.0},
        decay=0.7,
        max_hops=2,
    )
    assert scores.get("b", 0.0) > 0


def test_add_persists_supersedes_edge(tmp_path) -> None:
    db_path = tmp_path / "supersede.db"
    client = TestClient(
        create_app(
            Settings(
                database_path=str(db_path),
                auth_mode="none",
                embedding_provider="hashing",
                embedding_dim=64,
                organizer_provider="rule",
                decomposer_provider="off",
                reranker_provider="lexical",
            )
        )
    )
    with client:
        first = client.post(
            "/add",
            json={
                "request_id": "supersede-1",
                "user_id": "supersede-user",
                "session_id": "supersede-session",
                "messages": [
                    {"role": "user", "content": "I live in Beijing.", "timestamp": 1704067200000}
                ],
            },
        )
        second = client.post(
            "/add",
            json={
                "request_id": "supersede-2",
                "user_id": "supersede-user",
                "session_id": "supersede-session",
                "messages": [
                    {"role": "user", "content": "I live in Shanghai.", "timestamp": 1704153600000}
                ],
            },
        )
        assert first.status_code == 200
        assert second.status_code == 200

    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            "SELECT edge_type, COUNT(*) FROM memory_edges GROUP BY edge_type"
        ).fetchall()
    finally:
        connection.close()

    assert any(edge_type == "supersedes" for edge_type, _ in rows)
