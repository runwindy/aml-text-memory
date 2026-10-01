from __future__ import annotations

from app.ingestion.models import CanonicalMessage
from app.memory.models import MemoryRecord
from app.memory.multigranularity import build_session_summaries
from app.retrieval.granularity import granularity_scores
from app.retrieval.query_analyzer import QueryPlan
from app.schemas import AddRequest


def make_message(sequence_no: int, role: str, content: str) -> CanonicalMessage:
    return CanonicalMessage(
        message_id=f"msg-{sequence_no}",
        request_id="r1",
        user_id="u1",
        session_id="s1",
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


def test_build_session_summary() -> None:
    request = AddRequest(
        request_id="r1",
        messages=[{"role": "user", "content": "placeholder"}],
        user_id="u1",
        session_id="s1",
    )
    records = build_session_summaries(
        request=request,
        messages=[make_message(0, "user", "I live in Shanghai.")],
    )
    assert len(records) == 1
    assert records[0].memory_type == "summary"
    assert records[0].metadata["granularity"] == "session"


def test_granularity_scores_prefer_atomic_for_fact() -> None:
    atomic = MemoryRecord(id="a", user_id="u", session_id="s", content="a", metadata={"granularity": "atomic"})
    window = MemoryRecord(id="w", user_id="u", session_id="s", content="w", metadata={"granularity": "window"})
    session = MemoryRecord(id="s", user_id="u", session_id="s", content="s", metadata={"granularity": "session"})
    plan = QueryPlan(query_text="fact", retrieval_text="fact", query_type="fact", has_options=False)
    scores = granularity_scores([atomic, window, session], plan)
    assert scores["a"] > scores["w"] > scores["s"]
