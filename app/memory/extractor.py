from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Protocol

from app.memory.models import MemoryRecord
from app.schemas import AddMessage, AddRequest, ContentPart


class MemoryExtractor(Protocol):
    async def extract(self, request: AddRequest) -> list[MemoryRecord]:
        ...


def content_to_text(content: str | list[ContentPart]) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if part.type == "text":
            parts.append(part.text or "")
        elif part.type == "image_url" and part.image_url is not None:
            parts.append(f"[image:{part.image_url.url[:64]}]")
    return "\n".join(parts)


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
    index: int,
    content: str,
) -> str:
    raw = f"{user_id}\x1f{session_id}\x1f{request_id}\x1f{index}\x1f{content}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def render_message(message: AddMessage, index: int) -> str:
    text = content_to_text(message.content)
    timestamp = format_timestamp(message.timestamp)
    prefix_parts = [f"[{message.role}]"]
    if timestamp:
        prefix_parts.insert(0, f"[{timestamp}]")
    return " ".join(prefix_parts) + " " + text


class PassThroughExtractor:
    """Baseline extractor: one memory record per source message.

    This is intentionally simple and deterministic. Replace it with an LLM or
    rule-based extractor that returns atomic facts, events, preferences, and
    summaries.
    """

    def __init__(self, *, include_role: bool = True, include_timestamp: bool = True) -> None:
        self.include_role = include_role
        self.include_timestamp = include_timestamp

    async def extract(self, request: AddRequest) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []
        for index, message in enumerate(request.messages):
            text = content_to_text(message.content)
            if not text.strip():
                continue

            content_parts: list[str] = []
            if self.include_timestamp and message.timestamp is not None:
                formatted = format_timestamp(message.timestamp)
                if formatted:
                    content_parts.append(f"[{formatted}]")
            if self.include_role:
                content_parts.append(f"[{message.role}]")
            content_parts.append(text)
            content = " ".join(content_parts)

            records.append(
                MemoryRecord(
                    id=stable_memory_id(
                        user_id=request.user_id,
                        session_id=request.session_id,
                        request_id=request.request_id,
                        index=index,
                        content=content,
                    ),
                    user_id=request.user_id,
                    session_id=request.session_id,
                    request_id=request.request_id,
                    content=content,
                    memory_type="raw",
                    timestamp=message.timestamp,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    metadata={"source_index": index, "role": message.role},
                )
            )
        return records


def build_extractor() -> PassThroughExtractor:
    """Factory hook for future extractors."""

    return PassThroughExtractor()
