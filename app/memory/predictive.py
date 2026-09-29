from __future__ import annotations

import logging
from typing import Sequence

from app.memory.models import MemoryRecord
from app.retrieval.embedding import EmbeddingProvider
from app.retrieval.hybrid import MemoryHit, cosine_similarity

logger = logging.getLogger(__name__)

_MAX_TEXT_CHARS = 4000


def _clip(text: str) -> str:
    value = (text or "").strip()
    if len(value) <= _MAX_TEXT_CHARS:
        return value
    return value[:_MAX_TEXT_CHARS]


async def annotate_prediction_surprise(
    records: Sequence[MemoryRecord],
    embedder: EmbeddingProvider,
    *,
    surprise_threshold: float = 0.35,
) -> None:
    """JEPA-inspired surprise annotation.

    Context = user-side text in the window.
    Target  = assistant response in the window.

    A high cosine distance means the assistant reply is less predictable from
    the user context.  In this lightweight implementation that is a useful
    proxy for novelty / update / conflict, to be verified later by the
    deterministic or LLM update judge.
    """

    candidates = [
        record
        for record in records
        if record.metadata.get("assistant_target_text")
        and record.metadata.get("user_context_text")
    ]
    if not candidates:
        return

    contexts = [_clip(str(record.metadata.get("user_context_text", ""))) for record in candidates]
    targets = [_clip(str(record.metadata.get("assistant_target_text", ""))) for record in candidates]

    try:
        vectors = await embedder.embed(contexts + targets)
    except Exception:
        logger.warning("predictive embedding failed; skipping surprise annotation", exc_info=True)
        return

    if len(vectors) != len(contexts) + len(targets):
        logger.warning(
            "predictive embedding count mismatch: contexts=%d targets=%d vectors=%d",
            len(contexts),
            len(targets),
            len(vectors),
        )
        return

    context_vectors = vectors[: len(contexts)]
    target_vectors = vectors[len(contexts) :]

    for record, context_vector, target_vector in zip(candidates, context_vectors, target_vectors):
        similarity = cosine_similarity(context_vector, target_vector)
        surprise = max(0.0, min(1.0, 1.0 - similarity))
        signals = set(record.metadata.get("assistant_signals") or [])

        if "correction" in signals or "update" in signals:
            grounding_status = "assistant_update"
        elif surprise <= surprise_threshold:
            grounding_status = "assistant_consistent"
        else:
            grounding_status = "assistant_novel"

        record.metadata["prediction_error"] = round(surprise, 6)
        record.metadata["jepa_prediction_error"] = round(surprise, 6)
        record.metadata["grounding_status"] = grounding_status


def apply_predictive_bias(
    hits: Sequence[MemoryHit],
    weight: float = 0.15,
) -> list[MemoryHit]:
    """Apply a small JEPA-inspired trust adjustment after base reranking."""

    adjusted: list[MemoryHit] = []
    for hit in hits:
        metadata = hit.record.metadata or {}
        status = metadata.get("grounding_status")
        signals = set(metadata.get("assistant_signals") or [])
        adjustment = 0.0

        if status == "assistant_consistent":
            adjustment += weight
        elif status == "assistant_update":
            adjustment += weight * 0.5
        elif status == "assistant_novel":
            adjustment -= weight * 0.5

        if {"confirmation", "summary"} & signals:
            adjustment += weight * 0.5
        if "correction" in signals:
            adjustment += weight * 0.25

        adjusted.append(MemoryHit(record=hit.record, score=hit.score + adjustment))

    adjusted.sort(key=lambda item: item.score, reverse=True)
    return adjusted
