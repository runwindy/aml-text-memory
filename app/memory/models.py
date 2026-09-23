from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MemoryRecord:
    """One persisted memory unit returned by Search.

    `content` is what the platform Answer model receives. Keep it compact,
    self-contained, and evidence-focused.
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
