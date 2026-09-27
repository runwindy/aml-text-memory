from __future__ import annotations

from app.memory.models import MemoryRecord
from app.retrieval.conflict import apply_current_state_bias
from app.retrieval.result_window import expand_result_window
from app.retrieval.temporal_resolver import temporal_scores


def make_record(record_id: str, ordinal: int, content: str, timestamp: int) -> MemoryRecord:
    return MemoryRecord(
        id=record_id,
        user_id="u1",
        session_id="s1",
        content=content,
        memory_type="raw",
        timestamp=timestamp,
        metadata={"window_ordinal": ordinal, "chunk_type": "dialogue_window"},
    )


def test_result_window_expansion():
    records = [
        make_record("w0", 0, "window 0", 1000),
        make_record("w1", 1, "window 1", 2000),
        make_record("w2", 2, "window 2", 3000),
    ]
    fused = {"w0": 1.0, "w2": 0.1}
    expanded = expand_result_window(records, fused, ["w0"], window=1, seed_k=1)
    assert expanded["w1"] > 0.0


def test_temporal_scores_prefer_current_active():
    active = MemoryRecord(
        id="active",
        user_id="u1",
        session_id="s1",
        content="user lives in Shanghai",
        memory_type="fact",
        subject="user",
        predicate="live_in",
        object_value="Shanghai",
        status="active",
        valid_from="2024-01-01",
        timestamp=1710000000000,
    )
    superseded = MemoryRecord(
        id="old",
        user_id="u1",
        session_id="s1",
        content="user lived in Beijing",
        memory_type="fact",
        subject="user",
        predicate="live_in",
        object_value="Beijing",
        status="superseded",
        valid_from="2020-01-01",
        valid_to="2024-01-01",
        timestamp=1600000000000,
    )
    scores = temporal_scores([active, superseded], "Where does user live now?")
    assert scores["active"] > scores["old"]


def test_conflict_policy_current_and_past():
    active = MemoryRecord(
        id="active",
        user_id="u1",
        session_id="s1",
        content="current",
        memory_type="fact",
        subject="user",
        predicate="live_in",
        object_value="Shanghai",
        status="active",
        timestamp=1710000000000,
    )
    superseded = MemoryRecord(
        id="old",
        user_id="u1",
        session_id="s1",
        content="old",
        memory_type="fact",
        subject="user",
        predicate="live_in",
        object_value="Beijing",
        status="superseded",
        valid_to="2024-01-01",
        timestamp=1600000000000,
    )
    records = {"active": active, "old": superseded}
    current = apply_current_state_bias({"active": 1.0, "old": 1.0}, records, "Where now?")
    past = apply_current_state_bias({"active": 1.0, "old": 1.0}, records, "Where before?")
    assert current["active"] > current["old"]
    assert past["old"] > past["active"]
