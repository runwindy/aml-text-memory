from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Sequence

from app.ingestion.models import CanonicalMessage
from app.memory.models import MemoryRecord
from app.schemas import AddRequest


def _summary_id(*, user_id: str, session_id: str) -> str:
    raw = f"{user_id}\x1f{session_id}\x1fsession_summary"
    return "sum_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def build_session_summaries(
    *,
    request: AddRequest,
    messages: Sequence[CanonicalMessage],
    max_chars: int = 4000,
) -> list[MemoryRecord]:
    """Create deterministic extractive session summaries.

    This is intentionally not an LLM summary.  It gives the retrieval layer a
    global-context node that can be expanded back to atomic evidence via
    source_message_ids.
    """

    sessions: dict[str, list[CanonicalMessage]] = defaultdict(list)
    for message in messages:
        sessions[message.session_id].append(message)

    records: list[MemoryRecord] = []
    for session_id, session_messages in sessions.items():
        session_messages.sort(key=lambda item: item.sequence_no)
        lines = [
            f"[{message.role}] {message.raw_content}"
            for message in session_messages
            if message.raw_content.strip()
        ]
        if not lines:
            continue
        separator = "\n"
        content = "[session summary]" + separator + separator.join(lines)
        if max_chars > 0 and len(content) > max_chars:
            content = content[:max_chars].rstrip() + " ..."

        records.append(
            MemoryRecord(
                id=_summary_id(user_id=request.user_id, session_id=session_id),
                user_id=request.user_id,
                session_id=session_id,
                request_id=request.request_id,
                content=content,
                memory_type="summary",
                timestamp=session_messages[-1].timestamp_ms,
                created_at=session_messages[0].created_at,
                source_message_ids=[message.message_id for message in session_messages],
                confidence=0.6,
                importance=0.5,
                status="active",
                metadata={
                    "granularity": "session",
                    "summary_kind": "extractive",
                    "source_message_ids": [message.message_id for message in session_messages],
                    "message_count": len(session_messages),
                    "source_roles": list(dict.fromkeys(message.role for message in session_messages)),
                },
            )
        )
    return records
