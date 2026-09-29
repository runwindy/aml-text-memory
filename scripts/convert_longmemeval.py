"""Convert LongMemEval JSON into grouped local evaluation JSONL.

Supports streaming for the large M split.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import ijson


def parse_datetime(value: str | None) -> int | None:
    if not value:
        return None
    formats = (
        "%Y/%m/%d (%a) %H:%M",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d",
        "%Y-%m-%d",
    )
    for fmt in formats:
        try:
            parsed = datetime.strptime(value.strip(), fmt).replace(tzinfo=timezone.utc)
            return int(parsed.timestamp() * 1000)
        except ValueError:
            continue
    return None


def stable_keywords(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def iter_items(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("rb") as handle:
        yield from ijson.items(handle, "item")


def convert_item(item: dict[str, Any]) -> dict[str, Any]:
    question_id = str(item.get("question_id", ""))
    user_id = f"longmemeval:{question_id}"
    sessions = []
    session_ids = item.get("haystack_session_ids") or []
    session_dates = item.get("haystack_dates") or []
    haystack_sessions = item.get("haystack_sessions") or []

    for index, messages in enumerate(haystack_sessions):
        raw_session_id = str(session_ids[index]) if index < len(session_ids) else f"session-{index}"
        date_value = session_dates[index] if index < len(session_dates) else None
        timestamp = parse_datetime(date_value if isinstance(date_value, str) else None)
        converted_messages = []
        for message in messages:
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            converted: dict[str, Any] = {
                "role": str(message.get("role", "user")),
                "content": content,
            }
            if timestamp is not None:
                converted["timestamp"] = timestamp
            converted_messages.append(converted)

        if converted_messages:
            sessions.append(
                {
                    "session_id": f"{question_id}:{raw_session_id}",
                    "date_time": date_value,
                    "messages": converted_messages,
                }
            )

    evidence_texts: list[str] = []
    for messages in haystack_sessions:
        for message in messages:
            if message.get("has_answer"):
                text = str(message.get("content", "")).strip()
                if text:
                    evidence_texts.append(text)

    answer = item.get("answer")
    if isinstance(answer, list):
        gold_answers = [str(value) for value in answer]
    else:
        gold_answers = [str(answer)] if answer is not None else []

    expected_keywords = stable_keywords(evidence_texts or gold_answers)

    return {
        "id": question_id,
        "user_id": user_id,
        "sessions": sessions,
        "question": str(item.get("question", "")),
        "options": None,
        "expected_keywords": expected_keywords,
        "forbidden_keywords": [],
        "gold_answer": gold_answers,
        "category": str(item.get("question_type", "")),
        "question_date": item.get("question_date"),
        "answer_session_ids": item.get("answer_session_ids", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LongMemEval JSON to eval JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-items", type=int, default=None)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for item in iter_items(args.input):
            if args.max_items is not None and count >= args.max_items:
                break
            converted = convert_item(item)
            handle.write(json.dumps(converted, ensure_ascii=False) + "\n")
            count += 1
            if count % 100 == 0:
                print(f"converted={count}", flush=True)

    print(f"done count={count} output={output_path}", flush=True)


if __name__ == "__main__":
    main()
