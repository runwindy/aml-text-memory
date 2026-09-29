from __future__ import annotations

from types import SimpleNamespace

from app.memory.assistant import assistant_metadata, detect_assistant_signals
from app.memory.models import MemoryRecord
from app.memory.predictive import annotate_prediction_surprise, apply_predictive_bias
from app.retrieval.embedding import HashingEmbeddingProvider
from app.retrieval.hybrid import MemoryHit


def test_detect_assistant_signals() -> None:
    assert "confirmation" in detect_assistant_signals("Got it, understood.")
    assert "update" in detect_assistant_signals("Now you live in Shanghai.")
    assert "correction" in detect_assistant_signals("Actually, that is not correct.")


def test_assistant_metadata_marks_roles_and_signals() -> None:
    messages = [
        SimpleNamespace(role="user", raw_content="I moved to Shanghai.", message_id="u1"),
        SimpleNamespace(
            role="assistant",
            raw_content="Got it. You previously lived in Beijing, now you live in Shanghai.",
            message_id="a1",
        ),
    ]
    metadata = assistant_metadata(messages)  # type: ignore[arg-type]
    assert metadata["has_assistant"] is True
    assert metadata["source_roles"] == ["user", "assistant"]
    assert metadata["assistant_message_ids"] == ["a1"]
    assert "confirmation" in metadata["assistant_signals"]
    assert metadata["assistant_target_text"]


async def test_predictive_surprise_annotation() -> None:
    record = MemoryRecord(
        id="m1",
        user_id="u1",
        session_id="s1",
        content="user lives in Shanghai",
        metadata={
            "user_context_text": "I moved from Beijing to Shanghai.",
            "assistant_target_text": "Got it, now you live in Shanghai.",
            "assistant_signals": ["confirmation", "update"],
        },
    )
    embedder = HashingEmbeddingProvider(dimension=128)
    await annotate_prediction_surprise([record], embedder)
    assert "prediction_error" in record.metadata
    assert 0.0 <= float(record.metadata["prediction_error"]) <= 1.0
    assert record.metadata["grounding_status"] in {
        "assistant_consistent",
        "assistant_novel",
        "assistant_update",
    }


def test_predictive_bias_prefers_consistent_records() -> None:
    consistent = MemoryHit(
        record=MemoryRecord(
            id="consistent",
            user_id="u1",
            session_id="s1",
            content="consistent",
            metadata={"grounding_status": "assistant_consistent", "assistant_signals": ["confirmation"]},
        ),
        score=0.5,
    )
    novel = MemoryHit(
        record=MemoryRecord(
            id="novel",
            user_id="u1",
            session_id="s1",
            content="novel",
            metadata={"grounding_status": "assistant_novel", "assistant_signals": []},
        ),
        score=0.5,
    )
    adjusted = apply_predictive_bias([novel, consistent], weight=0.15)
    assert adjusted[0].record.id == "consistent"
