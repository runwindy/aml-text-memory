from __future__ import annotations

from dataclasses import replace

from app.ingestion.models import CanonicalMessage


def deduplicate(messages: list[CanonicalMessage]) -> list[CanonicalMessage]:
    """Mark exact duplicates without deleting the raw evidence.

    Bronze keeps the original payload. Silver keeps every message but marks
    exact duplicates so Gold extraction can skip them.
    """

    seen: set[tuple[str, str, str, str]] = set()
    result: list[CanonicalMessage] = []

    for message in messages:
        key = (message.user_id, message.session_id, message.role, message.content_hash)
        if key in seen:
            result.append(
                replace(
                    message,
                    quality_flags=[*message.quality_flags, "exact_duplicate"],
                )
            )
        else:
            seen.add(key)
            result.append(message)

    return result
