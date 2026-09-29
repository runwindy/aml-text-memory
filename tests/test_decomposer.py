from __future__ import annotations

from app.decomposer.graph_builder import build_memory_records
from app.decomposer.schemas import DecompositionResult, Event, Proposition, Relation
from app.ingestion.models import CanonicalMessage
from app.memory.window_extractor import DialogueWindow
from app.schemas import AddRequest


def make_message(index: int, content: str) -> CanonicalMessage:
    return CanonicalMessage(
        message_id=f"msg-{index}",
        request_id="req-1",
        user_id="user-1",
        session_id="session-1",
        sequence_no=index,
        role="user",
        raw_content=content,
        raw_content_hash=f"raw-hash-{index}",
        normalized_content=content.lower(),
        content_hash=f"hash-{index}",
        timestamp_ms=1704067200000 + index * 1000,
        timestamp_inferred=False,
        time_granularity=None,
        language="en",
    )


def test_decomposition_to_records():
    request = AddRequest(
        request_id="req-1",
        messages=[{"role": "user", "content": "I lived in Beijing and later moved to Shanghai."}],
        user_id="user-1",
        session_id="session-1",
    )
    messages = [
        make_message(0, "I lived in Beijing."),
        make_message(1, "Later I moved to Shanghai."),
    ]
    window = DialogueWindow(
        window_id="win-1",
        user_id="user-1",
        session_id="session-1",
        request_id="req-1",
        messages=messages,
        ordinal=0,
        start_sequence_no=0,
        end_sequence_no=1,
        start_timestamp_ms=1704067200000,
        end_timestamp_ms=1704067201000,
    )
    result = DecompositionResult(
        propositions=[
            Proposition(
                proposition_id="p1",
                memory_type="fact",
                subject="user",
                predicate="live_in",
                object_value="Beijing",
                valid_from="2023-01-01",
                valid_to="2024-06-01",
                confidence=0.9,
                source_message_indices=[0],
            ),
            Proposition(
                proposition_id="p2",
                memory_type="fact",
                subject="user",
                predicate="live_in",
                object_value="Shanghai",
                valid_from="2024-06-01",
                confidence=0.9,
                source_message_indices=[1],
            ),
        ],
        events=[
            Event(
                event_id="e1",
                event_type="moved_to",
                subject="user",
                to_value="Shanghai",
                timestamp_ms=1704067201000,
                confidence=0.9,
                source_message_indices=[1],
            )
        ],
        relations=[
            Relation(
                source="p2",
                target="e1",
                relation_type="causes_state",
                confidence=0.8,
                source_message_indices=[1],
            )
        ],
    )

    records = build_memory_records(request=request, window=window, result=result)
    assert {record.memory_type for record in records} == {"fact", "event", "relation"}
    fact_records = [record for record in records if record.memory_type == "fact"]
    assert fact_records[0].source_message_ids == ["msg-0"]
    assert fact_records[1].source_message_ids == ["msg-1"]
    relation = next(record for record in records if record.memory_type == "relation")
    p2_record = next(
        record for record in fact_records if record.object_value == "Shanghai"
    )
    event_record = next(record for record in records if record.memory_type == "event")
    assert relation.subject == p2_record.id
    assert relation.object_value == event_record.id
