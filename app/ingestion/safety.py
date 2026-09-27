from __future__ import annotations

import re
from dataclasses import replace

from app.ingestion.models import CanonicalMessage

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\-\s]{7,}\d)(?!\d)")
_ID_RE = re.compile(r"\b\d{15,18}[0-9Xx]\b")
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_PROMPT_INJECTION_RE = re.compile(
    r"(ignore\s+(all\s+)?(previous|prior)\s+instructions|"
    r"system\s+prompt|you\s+are\s+now|"
    r"忽略.{0,8}(之前|以上|前面).{0,8}(指令|要求)|"
    r"系统提示|请执行以下指令|把答案改成)",
    re.IGNORECASE,
)


def _append_unique(values: list[str], new_values: list[str]) -> list[str]:
    result = list(values)
    for value in new_values:
        if value not in result:
            result.append(value)
    return result


def process(messages: list[CanonicalMessage]) -> list[CanonicalMessage]:
    result: list[CanonicalMessage] = []

    for message in messages:
        text = message.raw_content
        pii_flags: list[str] = []
        safety_flags: list[str] = []

        if _EMAIL_RE.search(text):
            pii_flags.append("email")
        if _PHONE_RE.search(text):
            pii_flags.append("phone")
        if _ID_RE.search(text):
            pii_flags.append("government_id")
        if _IP_RE.search(text):
            pii_flags.append("ip_address")
        if _PROMPT_INJECTION_RE.search(text):
            safety_flags.append("prompt_injection_suspected")

        result.append(
            replace(
                message,
                pii_flags=_append_unique(message.pii_flags, pii_flags),
                safety_flags=_append_unique(message.safety_flags, safety_flags),
            )
        )

    return result
