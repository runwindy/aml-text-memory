from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.memory.models import MemoryRecord
from app.schemas import AddRequest


@dataclass(slots=True)
class RawAddRequest:
    request_id: str
    user_id: str
    session_id: str
    payload_json: str
    payload_hash: str
    received_at: str
    api_version: str = "v1"
    status: str = "received"
    cleaner_version: str = "cleaning-v1"


@dataclass(slots=True)
class CanonicalMessage:
    message_id: str
    request_id: str
    user_id: str
    session_id: str
    sequence_no: int
    role: str
    raw_content: str
    normalized_content: str
    content_hash: str
    timestamp_ms: int | None
    timestamp_inferred: bool
    time_granularity: str | None
    language: str
    pii_flags: list[str] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)
    safety_flags: list[str] = field(default_factory=list)
    ingestion_version: str = "cleaning-v1"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass(slots=True)
class IngestionResult:
    request: AddRequest
    raw: RawAddRequest
    messages: list[CanonicalMessage]
    memories: list[MemoryRecord]
    metadata: dict[str, Any] = field(default_factory=dict)
