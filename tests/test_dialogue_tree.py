from __future__ import annotations

from app.ingestion.models import CanonicalMessage
from app.memory.dialogue_tree import build_dialogue_tree
from app.schemas import AddRequest


def make_message(sequence_no: int, role: str, content: str) -> CanonicalMessage:
    return CanonicalMessage(
        message_id=f"msg-{sequence_no}",
        request_id="req-1",
        user_id="user-1",
        session_id="session-1",
        sequence_no=sequence_no,
        role=role,
        raw_content=content,
        raw_content_hash=f"raw-{sequence_no}",
        normalized_content=content,
        content_hash=f"hash-{sequence_no}",
        timestamp_ms=1704067200000 + sequence_no * 1000,
        timestamp_inferred=False,
        time_granularity=None,
        language="en",
    )


def test_build_dialogue_tree_orders_turns_and_messages() -> None:
    request = AddRequest(
        request_id="req-1",
        messages=[{"role": "user", "content": "placeholder"}],
        user_id="user-1",
        session_id="session-1",
    )
    messages = [
        make_message(0, "user", "I moved to Shanghai."),
        make_message(1, "assistant", "Got it, now you live in Shanghai."),
        make_message(2, "user", "Where did I live before?"),
        make_message(3, "assistant", "Previously you lived in Beijing."),
    ]
    nodes = build_dialogue_tree(request, messages)
    sessions = [node for node in nodes if node.node_type == "session"]
    turns = [node for node in nodes if node.node_type == "turn"]
    message_nodes = [node for node in nodes if node.node_type == "message"]

    assert len(sessions) == 1
    assert len(turns) == 2
    assert len(message_nodes) == 4
    assert turns[0].next_id == turns[1].node_id
    assert turns[1].prev_id == turns[0].node_id
    assert message_nodes[0].next_id == message_nodes[1].node_id
    assert message_nodes[1].prev_id == message_nodes[0].node_id
    assert message_nodes[1].metadata["assistant_signals"]

def test_add_persists_dialogue_tree_and_graph(tmp_path) -> None:
    import sqlite3

    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.main import create_app

    db_path = tmp_path / "tree.db"
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
        response = client.post(
            "/add",
            json={
                "request_id": "tree-request-1",
                "user_id": "tree-user",
                "session_id": "tree-session",
                "messages": [
                    {"role": "user", "content": "I moved to Shanghai.", "timestamp": 1704067200000},
                    {
                        "role": "assistant",
                        "content": "Got it, now you live in Shanghai.",
                        "timestamp": 1704067260000,
                    },
                ],
            },
        )
        assert response.status_code == 200

    connection = sqlite3.connect(db_path)
    try:
        dialogue_count = connection.execute("SELECT COUNT(*) FROM dialogue_nodes").fetchone()[0]
        edge_count = connection.execute("SELECT COUNT(*) FROM memory_edges").fetchone()[0]
        entity_count = connection.execute("SELECT COUNT(*) FROM entity_nodes").fetchone()[0]
        entity_link_count = connection.execute("SELECT COUNT(*) FROM memory_entity_links").fetchone()[0]
    finally:
        connection.close()

    assert dialogue_count >= 4
    assert edge_count >= 1
    assert entity_count >= 2
    assert entity_link_count >= 2

