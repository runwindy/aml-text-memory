from __future__ import annotations

import re
from dataclasses import replace

from app.ingestion.models import CanonicalMessage

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_MONTH_RE = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b",
    re.IGNORECASE,
)
_DATE_RE = re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b")
_CLOCK_RE = re.compile(r"\b\d{1,2}:\d{2}\b")


def infer_granularity(text: str) -> str | None:
    if _CLOCK_RE.search(text) or _DATE_RE.search(text):
        return "minute"
    if _MONTH_RE.search(text):
        return "month"
    if _YEAR_RE.search(text):
        return "year"
    return None


def process(messages: list[CanonicalMessage]) -> list[CanonicalMessage]:
    result: list[CanonicalMessage] = []
    for message in messages:
        result.append(
            replace(
                message,
                timestamp_inferred=message.timestamp_ms is None,
                time_granularity=infer_granularity(message.raw_content),
            )
        )
    return result
