from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Protocol, Sequence

from app.ingestion.models import CanonicalMessage
from app.memory.assistant import assistant_metadata
from app.memory.models import MemoryRecord
from app.schemas import AddRequest


class MemoryExtractor(Protocol):
    async def extract(
        self,
        request: AddRequest,
        messages: Sequence[CanonicalMessage],
    ) -> list[MemoryRecord]:
        ...


def format_timestamp(timestamp_ms: int | None) -> str:
    if timestamp_ms is None:
        return ""
    try:
        value = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        return value.isoformat()
    except (OverflowError, OSError, ValueError):
        return ""


def stable_memory_id(
    *,
    user_id: str,
    session_id: str,
    request_id: str,
    sequence_no: int,
    content: str,
) -> str:
    raw = f"{user_id}\x1f{session_id}\x1f{request_id}\x1f{sequence_no}\x1f{content}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class PassThroughExtractor:
    """Baseline extractor: one canonical message -> one memory record.

    The record keeps the raw text for Answer, while metadata carries the
    normalized retrieval text and ingestion flags.
    """

    def __init__(self, *, include_role: bool = True, include_timestamp: bool = True) -> None:
        self.include_role = include_role
        self.include_timestamp = include_timestamp

    async def extract(
        self,
        request: AddRequest,
        messages: Sequence[CanonicalMessage],
    ) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []

        for index, message in enumerate(messages):
            if not message.normalized_content:
                continue

            content_parts: list[str] = []
            if self.include_timestamp and message.timestamp_ms is not None:
                formatted = format_timestamp(message.timestamp_ms)
                if formatted:
                    content_parts.append(f"[{formatted}]")
            if self.include_role:
                content_parts.append(f"[{message.role}]")
            content_parts.append(message.raw_content)
            content = " ".join(content_parts)

            records.append(
                MemoryRecord(
                    id=stable_memory_id(
                        user_id=request.user_id,
                        session_id=request.session_id,
                        request_id=request.request_id,
                        sequence_no=message.sequence_no,
                        content=content,
                    ),
                    user_id=request.user_id,
                    session_id=request.session_id,
                    request_id=request.request_id,
                    content=content,
                    memory_type="raw",
                    timestamp=message.timestamp_ms,
                    created_at=message.created_at,
                    metadata={
                        "source_message_ids": [message.message_id],
                        "granularity": "message",
                        "normalized_content": message.normalized_content,
                        "language": message.language,
                        "time_granularity": message.time_granularity,
                        "timestamp_inferred": message.timestamp_inferred,
                        "pii_flags": message.pii_flags,
                        "quality_flags": message.quality_flags,
                        "safety_flags": message.safety_flags,
                        "time_mentions": message.time_mentions,
                        **assistant_metadata([message]),
                        "ingestion_version": message.ingestion_version,
                    },
                )
            )

        return records


def build_extractor() -> PassThroughExtractor:
    return PassThroughExtractor()
