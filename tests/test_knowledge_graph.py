from __future__ import annotations

from app.ingestion.models import CanonicalMessage
from app.memory.dialogue_tree import build_dialogue_tree
from app.memory.knowledge_graph import build_memory_edges
from app.memory.models import MemoryRecord
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


def test_build_memory_edges_includes_relation_and_assistant_support() -> None:
    request = AddRequest(
        request_id="req-1",
        messages=[{"role": "user", "content": "placeholder"}],
        user_id="user-1",
        session_id="session-1",
    )
    messages = [
        make_message(0, "user", "I moved to Shanghai."),
        make_message(1, "assistant", "Got it, now you live in Shanghai."),
    ]
    nodes = build_dialogue_tree(request, messages)
    relation_record = MemoryRecord(
        id="rel-1",
        user_id="user-1",
        session_id="session-1",
        content="relation",
        memory_type="relation",
        subject="mem-a",
        predicate="supersedes",
        object_value="mem-b",
        confidence=0.9,
    )
    edges = build_memory_edges(
        request=request,
        messages=messages,
        records=[relation_record],
        dialogue_nodes=nodes,
    )
    edge_types = {edge.edge_type for edge in edges}
    assert "supersedes" in edge_types
    assert "responds_to" in edge_types
