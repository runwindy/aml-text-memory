from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Sequence

from app.decomposer.schemas import DecompositionResult, Event, Proposition, Relation
from app.ingestion.models import CanonicalMessage
from app.memory.models import MemoryRecord
from app.memory.window_extractor import DialogueWindow
from app.schemas import AddRequest

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_predicate(predicate: str) -> str:
    value = _WHITESPACE_RE.sub("_", str(predicate or "").strip().lower())
    replacements = {
        "lives_in": "live_in",
        "live_at": "live_in",
        "works_at": "work_at",
        "work_for": "work_at",
        "likes": "likes",
        "loves": "likes",
        "prefers": "prefers",
        "hates": "dislikes",
        "dislikes": "dislikes",
    }
    return replacements.get(value, value or "related_to")


def _date_from_timestamp(timestamp_ms: int | None) -> str | None:
    if timestamp_ms is None:
        return None
    try:
        return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).date().isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def _source_ids(window: DialogueWindow, indices: Sequence[int]) -> list[str]:
    result: list[str] = []
    for index in indices:
        if 0 <= index < len(window.messages):
            message_id = window.messages[index].message_id
            if message_id not in result:
                result.append(message_id)
    return result


def _source_timestamp(window: DialogueWindow, indices: Sequence[int]) -> int | None:
    timestamps = [
        window.messages[index].timestamp_ms
        for index in indices
        if 0 <= index < len(window.messages) and window.messages[index].timestamp_ms is not None
    ]
    return max(timestamps) if timestamps else window.end_timestamp_ms


def _record_id(
    *,
    request: AddRequest,
    window: DialogueWindow,
    memory_type: str,
    subject: str,
    predicate: str,
    value: str,
    source_message_ids: Sequence[str],
) -> str:
    raw = (
        f"{request.user_id}\x1f{request.session_id}\x1f{window.window_id}\x1f"
        f"{memory_type}\x1f{subject}\x1f{predicate}\x1f{value}\x1f"
        + ",".join(source_message_ids)
    )
    return "mem_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _base_metadata(
    *,
    window: DialogueWindow,
    source_message_ids: Sequence[str],
    source_message_indices: Sequence[int],
    extra: dict | None = None,
) -> dict:
    metadata = {
        "decomposer": "llm-decomposer-v1",
        "source_message_ids": list(source_message_ids),
        "source_message_indices": list(source_message_indices),
        "window_id": window.window_id,
        "window_ordinal": window.ordinal,
    }
    if extra:
        metadata.update(extra)
    return metadata


def proposition_to_record(
    *,
    request: AddRequest,
    window: DialogueWindow,
    proposition: Proposition,
) -> MemoryRecord:
    source_message_ids = _source_ids(window, proposition.source_message_indices)
    timestamp_ms = proposition.timestamp_ms or _source_timestamp(
        window, proposition.source_message_indices
    )
    valid_from = proposition.valid_from or _date_from_timestamp(timestamp_ms)
    predicate = normalize_predicate(proposition.predicate)
    value = proposition.object_value
    content_parts = []
    if valid_from:
        content_parts.append(f"[{valid_from}]")
    if proposition.negated:
        content_parts.append("NOT")
    content_parts.append(f"{proposition.subject} {predicate} {value}.")
    content = " ".join(content_parts)

    return MemoryRecord(
        id=_record_id(
            request=request,
            window=window,
            memory_type=proposition.memory_type,
            subject=proposition.subject,
            predicate=predicate,
            value=value,
            source_message_ids=source_message_ids,
        ),
        user_id=request.user_id,
        session_id=request.session_id,
        request_id=request.request_id,
        content=content,
        memory_type=proposition.memory_type,
        timestamp=timestamp_ms,
        created_at=window.messages[0].created_at,
        subject=proposition.subject,
        predicate=predicate,
        object_value=value,
        entities=[proposition.subject, value],
        source_message_ids=source_message_ids,
        valid_from=valid_from,
        valid_to=proposition.valid_to,
        confidence=proposition.confidence,
        importance=0.7,
        status="active",
        metadata=_base_metadata(
            window=window,
            source_message_ids=source_message_ids,
            source_message_indices=proposition.source_message_indices,
            extra={
                "negated": proposition.negated,
                "modality": proposition.modality,
                "condition": proposition.condition,
                "time_text": proposition.time_text,
                "proposition_id": proposition.proposition_id,
            },
        ),
    )


def event_to_record(
    *,
    request: AddRequest,
    window: DialogueWindow,
    event: Event,
) -> MemoryRecord:
    source_message_ids = _source_ids(window, event.source_message_indices)
    timestamp_ms = event.timestamp_ms or _source_timestamp(window, event.source_message_indices)
    valid_from = _date_from_timestamp(timestamp_ms)
    value = event.to_value or event.object_value or event.event_type
    content = f"{event.subject} {event.event_type} {value}."

    return MemoryRecord(
        id=_record_id(
            request=request,
            window=window,
            memory_type="event",
            subject=event.subject,
            predicate=event.event_type,
            value=value,
            source_message_ids=source_message_ids,
        ),
        user_id=request.user_id,
        session_id=request.session_id,
        request_id=request.request_id,
        content=content,
        memory_type="event",
        timestamp=timestamp_ms,
        created_at=window.messages[0].created_at,
        subject=event.subject,
        predicate=event.event_type,
        object_value=value,
        entities=[event.subject, value],
        source_message_ids=source_message_ids,
        valid_from=valid_from,
        valid_to=None,
        confidence=event.confidence,
        importance=0.75,
        status="active",
        metadata=_base_metadata(
            window=window,
            source_message_ids=source_message_ids,
            source_message_indices=event.source_message_indices,
            extra={
                "event_id": event.event_id,
                "from_value": event.from_value,
                "to_value": event.to_value,
                "time_text": event.time_text,
            },
        ),
    )


def relation_to_record(
    *,
    request: AddRequest,
    window: DialogueWindow,
    relation: Relation,
    id_map: dict[str, str] | None = None,
) -> MemoryRecord:
    source_message_ids = _source_ids(window, relation.source_message_indices)
    predicate = normalize_predicate(relation.relation_type)
    id_map = id_map or {}
    source_id = id_map.get(relation.source, relation.source)
    target_id = id_map.get(relation.target, relation.target)
    content = f"[relation] {source_id} --{predicate}--> {target_id}"
    return MemoryRecord(
        id=_record_id(
            request=request,
            window=window,
            memory_type="relation",
            subject=source_id,
            predicate=predicate,
            value=target_id,
            source_message_ids=source_message_ids,
        ),
        user_id=request.user_id,
        session_id=request.session_id,
        request_id=request.request_id,
        content=content,
        memory_type="relation",
        timestamp=_source_timestamp(window, relation.source_message_indices),
        created_at=window.messages[0].created_at,
        subject=source_id,
        predicate=predicate,
        object_value=target_id,
        entities=[source_id, target_id],
        source_message_ids=source_message_ids,
        valid_from=None,
        valid_to=None,
        confidence=relation.confidence,
        importance=0.8,
        status="active",
        metadata=_base_metadata(
            window=window,
            source_message_ids=source_message_ids,
            source_message_indices=relation.source_message_indices,
            extra={"relation_type": predicate},
        ),
    )


def records_from_decomposition(
    *,
    request: AddRequest,
    window: DialogueWindow,
    result: DecompositionResult,
) -> list[MemoryRecord]:
    records: list[MemoryRecord] = []
    id_map: dict[str, str] = {}

    for proposition in result.propositions:
        record = proposition_to_record(
            request=request,
            window=window,
            proposition=proposition,
        )
        records.append(record)
        if proposition.proposition_id:
            id_map[proposition.proposition_id] = record.id

    for event in result.events:
        record = event_to_record(request=request, window=window, event=event)
        records.append(record)
        if event.event_id:
            id_map[event.event_id] = record.id

    for relation in result.relations:
        records.append(
            relation_to_record(
                request=request,
                window=window,
                relation=relation,
                id_map=id_map,
            )
        )
    return records
