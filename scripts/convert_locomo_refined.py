"""Convert LoCoMo-Refined public data into the local grouped evaluation format.

Output format: one JSON object per conversation.

{
  "id": "conv-26",
  "user_id": "locomo:conv-26",
  "sessions": [
    {
      "session_id": "conv-26:session_1",
      "messages": [
        {"role": "user", "content": "[Caroline] ...", "timestamp": 1683504000000}
      ]
    }
  ],
  "questions": [
    {
      "id": "conv-26#q0000",
      "question": "When did Caroline go to the LGBTQ support group?",
      "options": null,
      "expected_keywords": ["I went to a LGBTQ support group yesterday ..."],
      "forbidden_keywords": [],
      "gold_answer": ["7 May 2023"],
      "category": "2",
      "evidence": ["D1:3"]
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file = Path(path)
    return [
        json.loads(line)
        for line in file.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def parse_datetime(value: str | None) -> int | None:
    if not value:
        return None
    formats = (
        "%d %B, %Y",
        "%d %B %Y",
        "%d %b, %Y",
        "%d %b %Y",
        "%B %d, %Y",
        "%b %d, %Y",
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


def build_evidence_map(conversation: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    index = 1
    while f"session_{index}" in conversation:
        for message in conversation.get(f"session_{index}", []):
            dia_id = str(message.get("dia_id", "")).strip()
            text = str(message.get("text", "")).strip()
            if dia_id and text:
                result[dia_id] = text
        index += 1
    return result


def convert_conversation(
    item: dict[str, Any],
    questions: list[dict[str, Any]],
    *,
    include_answers_as_keywords: bool,
    max_questions: int | None,
) -> dict[str, Any]:
    sample_id = str(item["sample_id"])
    conversation = item["conversation"]
    speaker_a = str(conversation.get("speaker_a", "speaker 1"))
    speaker_b = str(conversation.get("speaker_b", "speaker 2"))
    evidence_map = build_evidence_map(conversation)

    sessions: list[dict[str, Any]] = []
    index = 1
    while f"session_{index}" in conversation:
        session_key = f"session_{index}"
        session_date = conversation.get(f"{session_key}_date_time")
        timestamp = parse_datetime(session_date)
        messages: list[dict[str, Any]] = []

        for message in conversation.get(session_key, []):
            speaker = str(message.get("speaker", "")).strip()
            text = str(message.get("text", "")).strip()
            if not text:
                continue
            role = "user" if speaker == speaker_a else "assistant"
            content = f"[{speaker}] {text}" if speaker else text
            record: dict[str, Any] = {"role": role, "content": content}
            if timestamp is not None:
                record["timestamp"] = timestamp
            messages.append(record)

        sessions.append(
            {
                "session_id": f"{sample_id}:{session_key}",
                "date_time": session_date,
                "messages": messages,
            }
        )
        index += 1

    converted_questions: list[dict[str, Any]] = []
    for question in questions[:max_questions] if max_questions else questions:
        evidence_ids = [str(value) for value in question.get("evidence", [])]
        evidence_texts: list[str] = []

        for evidence_message in question.get("evidence_messages", []) or []:
            text = str(evidence_message.get("text", "")).strip()
            if text:
                evidence_texts.append(text)

        if not evidence_texts:
            evidence_texts = [
                evidence_map[evidence_id]
                for evidence_id in evidence_ids
                if evidence_id in evidence_map
            ]

        expected = stable_keywords(evidence_texts)
        if include_answers_as_keywords:
            expected = stable_keywords(expected + [str(value) for value in question.get("answer", [])])

        converted_questions.append(
            {
                "id": str(question["qa_id"]),
                "question": str(question["question"]),
                "options": None,
                "expected_keywords": expected,
                "forbidden_keywords": [],
                "gold_answer": [str(value) for value in question.get("answer", [])],
                "category": str(question.get("category", "")),
                "evidence": evidence_ids,
                "evidence_texts": evidence_texts,
            }
        )

    return {
        "id": sample_id,
        "user_id": f"locomo:{sample_id}",
        "sessions": sessions,
        "questions": converted_questions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LoCoMo-Refined to grouped eval JSONL.")
    parser.add_argument(
        "--raw",
        default="datasets/locomo_refined/locomo_refined.json",
        help="Path to locomo_refined.json",
    )
    parser.add_argument(
        "--questions",
        default="datasets/locomo_refined/questions.jsonl",
        help="Path to questions.jsonl",
    )
    parser.add_argument(
        "--output",
        default="examples/locomo_refined_eval.jsonl",
        help="Output grouped eval JSONL",
    )
    parser.add_argument("--max-conversations", type=int, default=None)
    parser.add_argument("--max-questions-per-conversation", type=int, default=None)
    parser.add_argument(
        "--include-answers-as-keywords",
        action="store_true",
        help="Also use gold answers as local retrieval keywords.",
    )
    args = parser.parse_args()

    raw_items = json.loads(Path(args.raw).read_text(encoding="utf-8-sig"))
    question_rows = read_jsonl(args.questions)

    questions_by_sample: dict[str, list[dict[str, Any]]] = {}
    for question in question_rows:
        sample_id = str(question.get("sample_id", ""))
        questions_by_sample.setdefault(sample_id, []).append(question)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted_count = 0
    question_count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for item in raw_items:
            if args.max_conversations is not None and converted_count >= args.max_conversations:
                break
            sample_id = str(item.get("sample_id", ""))
            questions = questions_by_sample.get(sample_id, [])
            if not questions:
                continue
            converted = convert_conversation(
                item,
                questions,
                include_answers_as_keywords=args.include_answers_as_keywords,
                max_questions=args.max_questions_per_conversation,
            )
            handle.write(json.dumps(converted, ensure_ascii=False) + "\n")
            converted_count += 1
            question_count += len(converted["questions"])

    print(f"conversations={converted_count} questions={question_count} output={output_path}")


if __name__ == "__main__":
    main()

