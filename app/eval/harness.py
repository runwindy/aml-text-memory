from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx

from app.eval.metrics import average_metrics, evaluate_keyword_retrieval


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file = Path(path)
    with file.open("r", encoding="utf-8-sig") as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def auth_headers(key: str | None) -> dict[str, str]:
    if not key:
        return {}
    return {"X-Api-Key": key}


def add_session(
    client: httpx.Client,
    *,
    case_id: str,
    user_id: str,
    session: dict[str, Any],
) -> None:
    payload = {
        "request_id": f"eval:{case_id}:{session['session_id']}",
        "messages": session["messages"],
        "user_id": user_id,
        "session_id": session["session_id"],
    }

    retryable = {408, 409, 425, 429, 500, 502, 503, 504, 524}
    last_transport_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            response = client.post("/add", json=payload)
        except httpx.TransportError as exc:
            last_transport_error = exc
            if attempt >= 5:
                raise
            time.sleep(min(60.0, 2.0 ** attempt))
            continue

        if response.status_code in retryable and attempt < 5:
            retry_after = response.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else 2.0 ** attempt
            except ValueError:
                delay = 2.0 ** attempt
            time.sleep(min(60.0, max(1.0, delay)))
            continue

        response.raise_for_status()
        return

    if last_transport_error is not None:
        raise last_transport_error
    raise RuntimeError("add_session retry loop exhausted")


def search_case(
    client: httpx.Client,
    *,
    user_id: str,
    question: str,
    options: list[str] | None,
    top_k: int,
) -> list[dict[str, Any]]:
    payload: dict[str, Any] = {
        "query": question,
        "user_id": user_id,
        "top_k": top_k,
    }
    if options:
        payload["options"] = options

    retryable = {408, 425, 429, 500, 502, 503, 504, 524}
    last_transport_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            response = client.post("/search", json=payload)
        except httpx.TransportError as exc:
            last_transport_error = exc
            if attempt >= 4:
                raise
            time.sleep(min(60.0, 2.0 ** attempt))
            continue

        if response.status_code in retryable and attempt < 4:
            retry_after = response.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else 2.0 ** attempt
            except ValueError:
                delay = 2.0 ** attempt
            time.sleep(min(60.0, max(1.0, delay)))
            continue

        response.raise_for_status()
        return response.json()["data"]

    if last_transport_error is not None:
        raise last_transport_error
    raise RuntimeError("search_case retry loop exhausted")


def evaluate_question(
    client: httpx.Client,
    *,
    user_id: str,
    question: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    results = search_case(
        client,
        user_id=user_id,
        question=str(question["question"]),
        options=question.get("options"),
        top_k=top_k,
    )
    retrieved_texts = [str(item.get("content", "")) for item in results]
    metrics = evaluate_keyword_retrieval(
        retrieved_texts,
        question.get("expected_keywords", []),
        forbidden_keywords=question.get("forbidden_keywords", []),
    )
    return {
        "id": question["id"],
        "question": question["question"],
        "retrieved": results,
        "metrics": metrics,
    }


def evaluate_case(
    client: httpx.Client,
    case: dict[str, Any],
    *,
    top_k: int,
) -> dict[str, Any]:
    case_id = str(case["id"])
    user_id = str(case["user_id"])

    for session in case.get("sessions", []):
        add_session(client, case_id=case_id, user_id=user_id, session=session)

    row = evaluate_question(
        client,
        user_id=user_id,
        question=case,
        top_k=top_k,
    )
    row["group_id"] = case_id
    return row


def evaluate_group(
    client: httpx.Client,
    case: dict[str, Any],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """Evaluate one conversation with many questions.

    Sessions are added exactly once. This is the format used by long-conversation
    benchmarks such as LoCoMo-Refined.
    """

    case_id = str(case["id"])
    user_id = str(case["user_id"])

    for session in case.get("sessions", []):
        add_session(client, case_id=case_id, user_id=user_id, session=session)

    rows: list[dict[str, Any]] = []
    for question in case.get("questions", []):
        row = evaluate_question(
            client,
            user_id=user_id,
            question=question,
            top_k=top_k,
        )
        row["group_id"] = case_id
        rows.append(row)
    return rows


def evaluate_dataset(
    client: httpx.Client,
    cases: list[dict[str, Any]],
    *,
    top_k: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float]] = []

    for case in cases:
        if "questions" in case and "question" not in case:
            case_rows = evaluate_group(client, case, top_k=top_k)
        else:
            case_rows = [evaluate_case(client, case, top_k=top_k)]

        for row in case_rows:
            rows.append(row)
            metric_rows.append(row["metrics"])

    return {
        "average": average_metrics(metric_rows),
        "cases": rows,
    }
