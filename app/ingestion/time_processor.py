from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from app.ingestion.models import CanonicalMessage

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_MONTH_RE = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b",
    re.IGNORECASE,
)
_DATE_RE = re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b")
_CLOCK_RE = re.compile(r"\b\d{1,2}:\d{2}\b")

_GRANULARITY_RANK = {
    "year": 1,
    "month": 2,
    "day": 3,
    "minute": 4,
}


def extract_time_mentions(text: str) -> list[dict[str, Any]]:
    """Extract concrete time mentions without rewriting the source text.

    A message can contain multiple time expressions.  Each mention keeps its
    original text and character span so later extraction can bind an event to
    the exact evidence, while the message timestamp remains a separate field.
    """

    candidates: list[dict[str, Any]] = []
    for regex, granularity in (
        (_CLOCK_RE, "minute"),
        (_DATE_RE, "day"),
        (_MONTH_RE, "month"),
        (_YEAR_RE, "year"),
    ):
        for match in regex.finditer(text):
            candidates.append(
                {
                    "text": match.group(0),
                    "span_start": match.start(),
                    "span_end": match.end(),
                    "granularity": granularity,
                }
            )

    # Prefer the longest / most specific match when spans overlap.
    candidates.sort(key=lambda item: (item["span_start"], -(item["span_end"] - item["span_start"])))
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        overlaps = any(
            candidate["span_start"] < existing["span_end"]
            and existing["span_start"] < candidate["span_end"]
            for existing in selected
        )
        if not overlaps:
            selected.append(candidate)
    return sorted(selected, key=lambda item: item["span_start"])


def infer_granularity(text: str) -> str | None:
    mentions = extract_time_mentions(text)
    if not mentions:
        return None
    return max(
        (mention["granularity"] for mention in mentions),
        key=lambda granularity: _GRANULARITY_RANK.get(granularity, 0),
    )


def process(messages: list[CanonicalMessage]) -> list[CanonicalMessage]:
    result: list[CanonicalMessage] = []
    for message in messages:
        mentions = extract_time_mentions(message.raw_content)
        granularity = infer_granularity(message.raw_content)
        result.append(
            replace(
                message,
                timestamp_inferred=message.timestamp_ms is None,
                time_granularity=granularity,
                time_mentions=mentions,
            )
        )
    return result
