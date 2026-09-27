from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from app.ingestion.models import CanonicalMessage
from app.memory.models import MemoryRecord
from app.schemas import AddRequest


def _format_timestamp(timestamp_ms: int | None) -> str:
    if timestamp_ms is None:
        return ""
    try:
        return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return ""


def _window_id(
    *,
    user_id: str,
    session_id: str,
    messages: Sequence[CanonicalMessage],
    ordinal: int,
) -> str:
    raw = (
        f"{user_id}\x1f{session_id}\x1f{ordinal}\x1f"
        + ",".join(message.message_id for message in messages)
    )
    return "win_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


@dataclass(slots=True)
class DialogueWindow:
    window_id: str
    user_id: str
    session_id: str
    request_id: str
    messages: list[CanonicalMessage]
    ordinal: int
    start_sequence_no: int
    end_sequence_no: int
    start_timestamp_ms: int | None
    end_timestamp_ms: int | None

    @property
    def source_message_ids(self) -> list[str]:
        return [message.message_id for message in self.messages]

    def render_content(self) -> str:
        lines: list[str] = []
        for message in self.messages:
            timestamp = _format_timestamp(message.timestamp_ms)
            prefix = f"[{timestamp}] " if timestamp else ""
            lines.append(f"{prefix}[{message.role}] {message.raw_content}")
        return "\n".join(lines)

    def render_normalized(self) -> str:
        return "\n".join(
            f"[{message.role}] {message.normalized_content}"
            for message in self.messages
            if message.normalized_content
        )

    def to_memory_record(self) -> MemoryRecord:
        return MemoryRecord(
            id=self.window_id,
            user_id=self.user_id,
            session_id=self.session_id,
            request_id=self.request_id,
            content=self.render_content(),
            memory_type="raw",
            timestamp=self.end_timestamp_ms or self.start_timestamp_ms,
            created_at=self.messages[0].created_at,
            metadata={
                "chunk_type": "dialogue_window",
                "window_ordinal": self.ordinal,
                "window_size": len(self.messages),
                "start_sequence_no": self.start_sequence_no,
                "end_sequence_no": self.end_sequence_no,
                "start_timestamp_ms": self.start_timestamp_ms,
                "end_timestamp_ms": self.end_timestamp_ms,
                "source_message_ids": self.source_message_ids,
                "normalized_content": self.render_normalized(),
            },
        )


class DialogueWindowExtractor:
    """Build overlapping dialogue windows inside each session.

    The window is the retrieval/extraction unit. Raw messages remain in Silver.
    """

    def __init__(self, window_size: int = 3, overlap: int = 1) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if overlap < 0 or overlap >= window_size:
            raise ValueError("overlap must be >= 0 and smaller than window_size")
        self.window_size = window_size
        self.overlap = overlap
        self.step = window_size - overlap

    def build_windows(
        self,
        request: AddRequest,
        messages: Sequence[CanonicalMessage],
    ) -> list[DialogueWindow]:
        sessions: dict[str, list[CanonicalMessage]] = {}
        for message in messages:
            sessions.setdefault(message.session_id, []).append(message)

        windows: list[DialogueWindow] = []
        for session_id, session_messages in sessions.items():
            session_messages = sorted(session_messages, key=lambda item: item.sequence_no)
            ordinal = 0
            start = 0
            while start < len(session_messages):
                chunk = session_messages[start : start + self.window_size]
                if not chunk:
                    break
                windows.append(
                    DialogueWindow(
                        window_id=_window_id(
                            user_id=request.user_id,
                            session_id=session_id,
                            messages=chunk,
                            ordinal=ordinal,
                        ),
                        user_id=request.user_id,
                        session_id=session_id,
                        request_id=request.request_id,
                        messages=list(chunk),
                        ordinal=ordinal,
                        start_sequence_no=chunk[0].sequence_no,
                        end_sequence_no=chunk[-1].sequence_no,
                        start_timestamp_ms=chunk[0].timestamp_ms,
                        end_timestamp_ms=chunk[-1].timestamp_ms,
                    )
                )
                ordinal += 1
                if start + self.window_size >= len(session_messages):
                    break
                start += self.step
        return windows

    async def extract(
        self,
        request: AddRequest,
        messages: Sequence[CanonicalMessage],
    ) -> list[MemoryRecord]:
        return [
            window.to_memory_record()
            for window in self.build_windows(request, messages)
        ]
