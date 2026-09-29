from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Sequence

from app.ingestion.models import CanonicalMessage
from app.memory.assistant import assistant_metadata
from app.memory.models import MemoryRecord
from app.schemas import AddRequest

_NAME_PATTERNS = (
    re.compile(r"\bmy name is\s+([A-Z][A-Za-z'\-]+)", re.IGNORECASE),
    re.compile(r"\bi am\s+([A-Z][A-Za-z'\-]+)", re.IGNORECASE),
    re.compile(r"\bi'm\s+([A-Z][A-Za-z'\-]+)", re.IGNORECASE),
    re.compile(r"我叫([\u4e00-\u9fff]{2,4})"),
)
_LOCATION_PATTERNS = (
    re.compile(r"\bi live in\s+([^.,;\n]+)", re.IGNORECASE),
    re.compile(r"\bi moved to\s+([^.,;\n]+)", re.IGNORECASE),
    re.compile(r"我住在([\u4e00-\u9fffA-Za-z]+)"),
    re.compile(r"我搬到了([\u4e00-\u9fffA-Za-z]+)"),
)
_JOB_PATTERNS = (
    re.compile(r"\bi work (?:at|for)\s+([^.,;\n]+)", re.IGNORECASE),
    re.compile(r"\bi work as\s+(?:a|an)?\s*([^.,;\n]+)", re.IGNORECASE),
    re.compile(r"我在([\u4e00-\u9fff]+)工作"),
)
_PREFERENCE_PATTERNS = (
    re.compile(r"\bi (?:like|love|prefer)\s+([^.,;\n]+)", re.IGNORECASE),
    re.compile(r"我喜欢([^。\n]+)"),
    re.compile(r"我偏好([^。\n]+)"),
)
_DISLIKE_PATTERNS = (
    re.compile(r"\bi (?:hate|dislike|don't like)\s+([^.,;\n]+)", re.IGNORECASE),
    re.compile(r"我讨厌([^。\n]+)"),
    re.compile(r"我不喜欢([^。\n]+)"),
)
_THIRD_PERSON_PREFERENCE_PATTERNS = (
    (re.compile(r"\b([A-Z][A-Za-z'\-]+)\s+(?:likes|loves|prefers)\s+([^.,;\n]+)")),
    (re.compile(r"([\u4e00-\u9fff]{2,4})喜欢([^。\n]+)")),
)
_EVENT_PATTERNS = (
    (re.compile(r"\bi moved to\s+([^.,;\n]+)", re.IGNORECASE), "moved_to"),
    (re.compile(r"\bi visited\s+([^.,;\n]+)", re.IGNORECASE), "visited"),
    (re.compile(r"我搬到了([^。\n]+)"), "moved_to"),
    (re.compile(r"我去了([^。\n]+)"), "visited"),
)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\-\s]{7,}\d)(?!\d)")


def _date_from_timestamp(timestamp_ms: int | None) -> str | None:
    if timestamp_ms is None:
        return None
    try:
        return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).date().isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def _record_id(
    *,
    message: CanonicalMessage,
    memory_type: str,
    predicate: str,
    value: str,
) -> str:
    raw = (
        f"{message.user_id}\x1f{message.session_id}\x1f{message.message_id}\x1f"
        f"{memory_type}\x1f{predicate}\x1f{value}"
    )
    return "mem_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


class RuleBasedOrganizer:
    """Small deterministic organizer for the first structured Gold version.

    It extracts simple facts, preferences, profile fields and events. It is
    intentionally conservative: it only emits a structured memory when a
    high-confidence pattern matches.
    """

    def _make_record(
        self,
        *,
        request: AddRequest,
        message: CanonicalMessage,
        memory_type: str,
        subject: str,
        predicate: str,
        value: str,
        confidence: float,
    ) -> MemoryRecord:
        date = _date_from_timestamp(message.timestamp_ms)
        prefix = f"[{date}] " if date else ""
        content = f"{prefix}{subject} {predicate} {value}."
        return MemoryRecord(
            id=_record_id(
                message=message,
                memory_type=memory_type,
                predicate=predicate,
                value=value,
            ),
            user_id=request.user_id,
            session_id=request.session_id,
            request_id=request.request_id,
            content=content,
            memory_type=memory_type,
            timestamp=message.timestamp_ms,
            created_at=message.created_at,
            subject=subject,
            predicate=predicate,
            object_value=value,
            qualifiers={},
            entities=[subject, value],
            source_message_ids=[message.message_id],
            valid_from=date,
            valid_to=None,
            confidence=confidence,
            importance=0.7,
            status="active",
            metadata={
                "source_message_ids": [message.message_id],
                "normalized_content": message.normalized_content,
                "language": message.language,
                "time_granularity": message.time_granularity,
                "pii_flags": message.pii_flags,
                "quality_flags": message.quality_flags,
                "safety_flags": message.safety_flags,
                "time_mentions": message.time_mentions,
                **assistant_metadata([message]),
                "ingestion_version": message.ingestion_version,
                "organizer": "rule-based-v1",
            },
        )

    async def extract(
        self,
        request: AddRequest,
        messages: Sequence[CanonicalMessage],
    ) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []

        for message in messages:
            text = message.raw_content
            if not text.strip():
                continue

            subject = message.role

            for pattern in _NAME_PATTERNS:
                match = pattern.search(text)
                if match:
                    records.append(
                        self._make_record(
                            request=request,
                            message=message,
                            memory_type="profile",
                            subject=subject,
                            predicate="name",
                            value=match.group(1).strip(),
                            confidence=0.85,
                        )
                    )

            for pattern in _LOCATION_PATTERNS:
                match = pattern.search(text)
                if match:
                    records.append(
                        self._make_record(
                            request=request,
                            message=message,
                            memory_type="fact",
                            subject=subject,
                            predicate="location",
                            value=match.group(1).strip(),
                            confidence=0.8,
                        )
                    )

            for pattern in _JOB_PATTERNS:
                match = pattern.search(text)
                if match:
                    records.append(
                        self._make_record(
                            request=request,
                            message=message,
                            memory_type="fact",
                            subject=subject,
                            predicate="job",
                            value=match.group(1).strip(),
                            confidence=0.8,
                        )
                    )

            for pattern in _THIRD_PERSON_PREFERENCE_PATTERNS:
                match = pattern.search(text)
                if match:
                    records.append(
                        self._make_record(
                            request=request,
                            message=message,
                            memory_type="preference",
                            subject=match.group(1).strip(),
                            predicate="likes",
                            value=match.group(2).strip(),
                            confidence=0.75,
                        )
                    )

            for pattern, predicate in _EVENT_PATTERNS:
                match = pattern.search(text)
                if match:
                    records.append(
                        self._make_record(
                            request=request,
                            message=message,
                            memory_type="event",
                            subject=subject,
                            predicate=predicate,
                            value=match.group(1).strip(),
                            confidence=0.75,
                        )
                    )

            for pattern in _PREFERENCE_PATTERNS:
                match = pattern.search(text)
                if match:
                    records.append(
                        self._make_record(
                            request=request,
                            message=message,
                            memory_type="preference",
                            subject=subject,
                            predicate="likes",
                            value=match.group(1).strip(),
                            confidence=0.75,
                        )
                    )

            for pattern in _DISLIKE_PATTERNS:
                match = pattern.search(text)
                if match:
                    records.append(
                        self._make_record(
                            request=request,
                            message=message,
                            memory_type="preference",
                            subject=subject,
                            predicate="dislikes",
                            value=match.group(1).strip(),
                            confidence=0.75,
                        )
                    )

            email = _EMAIL_RE.search(text)
            if email:
                records.append(
                    self._make_record(
                        request=request,
                        message=message,
                        memory_type="profile",
                        subject=subject,
                        predicate="email",
                        value=email.group(0),
                        confidence=0.95,
                    )
                )

            phone = _PHONE_RE.search(text)
            if phone:
                records.append(
                    self._make_record(
                        request=request,
                        message=message,
                        memory_type="profile",
                        subject=subject,
                        predicate="phone",
                        value=phone.group(0),
                        confidence=0.9,
                    )
                )

        return records
