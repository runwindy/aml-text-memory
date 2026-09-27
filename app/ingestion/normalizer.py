from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from datetime import datetime, timezone

from app.ingestion.models import CanonicalMessage
from app.ingestion.text_utils import content_to_text
from app.schemas import AddRequest

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200d\u2060\ufeff]")
_MULTISPACE_RE = re.compile(r"[ \t]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def normalize_text(text: str) -> str:
    """Non-destructive normalization for retrieval.

    raw_content is never overwritten. This normalized form is used only for
    indexing, deduplication, and BM25.
    """

    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _CONTROL_RE.sub("", text)
    text = _MULTISPACE_RE.sub(" ", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def detect_language(text: str) -> str:
    has_cjk = bool(_CJK_RE.search(text))
    has_latin = bool(_LATIN_RE.search(text))
    if has_cjk and has_latin:
        return "mixed"
    if has_cjk:
        return "zh"
    if has_latin:
        return "en"
    return "unknown"


def stable_message_id(
    *,
    user_id: str,
    session_id: str,
    request_id: str,
    sequence_no: int,
    normalized_content: str,
) -> str:
    raw = f"{user_id}\x1f{session_id}\x1f{request_id}\x1f{sequence_no}\x1f{normalized_content}"
    return "msg_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def content_hash(
    *,
    user_id: str,
    session_id: str,
    role: str,
    normalized_content: str,
) -> str:
    raw = f"{user_id}\x1f{session_id}\x1f{role}\x1f{normalized_content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def normalize_messages(request: AddRequest) -> list[CanonicalMessage]:
    created_at = datetime.now(timezone.utc).isoformat()
    messages: list[CanonicalMessage] = []

    for sequence_no, message in enumerate(request.messages):
        raw_content = content_to_text(message.content)
        normalized = normalize_text(raw_content)
        flags: list[str] = []

        if not normalized:
            flags.append("empty_after_normalization")
        if len(normalized) > 20_000:
            flags.append("too_long")
        if message.timestamp is None:
            flags.append("missing_timestamp")

        messages.append(
            CanonicalMessage(
                message_id=stable_message_id(
                    user_id=request.user_id,
                    session_id=request.session_id,
                    request_id=request.request_id,
                    sequence_no=sequence_no,
                    normalized_content=normalized,
                ),
                request_id=request.request_id,
                user_id=request.user_id,
                session_id=request.session_id,
                sequence_no=sequence_no,
                role=message.role,
                raw_content=raw_content,
                normalized_content=normalized,
                content_hash=content_hash(
                    user_id=request.user_id,
                    session_id=request.session_id,
                    role=message.role,
                    normalized_content=normalized,
                ),
                timestamp_ms=message.timestamp,
                timestamp_inferred=message.timestamp is None,
                time_granularity=None,
                language=detect_language(normalized),
                quality_flags=flags,
                created_at=created_at,
            )
        )

    return messages
