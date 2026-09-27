from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MemoryRecord:
    """One persisted memory unit returned by Search.

    Raw records keep the original evidence text. Structured records carry
    subject/predicate/object fields and provenance so the retrieval layer can
    support facts, events, preferences, profiles, rules, summaries and
    relations.
    """

    id: str
    user_id: str
    session_id: str
    content: str
    memory_type: str = "raw"
    request_id: str | None = None
    timestamp: int | None = None
    created_at: str = ""
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Structured Gold fields.
    subject: str | None = None
    predicate: str | None = None
    object_value: str | None = None
    qualifiers: dict[str, Any] = field(default_factory=dict)
    entities: list[str] = field(default_factory=list)
    source_message_ids: list[str] = field(default_factory=list)
    valid_from: str | None = None
    valid_to: str | None = None
    confidence: float | None = None
    importance: float | None = None
    status: str = "active"
