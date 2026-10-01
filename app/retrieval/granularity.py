from __future__ import annotations

from typing import Sequence

from app.memory.models import MemoryRecord
from app.retrieval.query_analyzer import QueryPlan


def granularity_scores(
    records: Sequence[MemoryRecord],
    plan: QueryPlan,
    *,
    atomic_weight: float = 1.0,
    window_weight: float = 0.7,
    session_weight: float = 0.6,
    multi_session_weight: float = 0.9,
) -> dict[str, float]:
    """Prefer atomic evidence for factual queries and session summaries for
    broad multi-session questions."""

    broad = plan.query_type in {"multi_hop", "temporal"}
    scores: dict[str, float] = {}
    for record in records:
        granularity = str(record.metadata.get("granularity") or "")
        if granularity in {"atomic", "message"}:
            score = atomic_weight
        elif granularity == "window":
            score = window_weight
        elif granularity == "session":
            score = multi_session_weight if broad else session_weight
        else:
            score = 0.5
        scores[record.id] = score
    return scores
