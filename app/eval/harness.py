from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from app.eval.metrics import average_metrics, evaluate_keyword_retrieval


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file = Path(path)
    return [
        json.loads(line)
        for line in file.read_text(encoding="utf-8").splitlines()
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
    response = client.post("/add", json=payload)
    response.raise_for_status()


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
    response = client.post("/search", json=payload)
    response.raise_for_status()
    return response.json()["data"]


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

    results = search_case(
        client,
        user_id=user_id,
        question=str(case["question"]),
        options=case.get("options"),
        top_k=top_k,
    )
    retrieved_texts = [str(item.get("content", "")) for item in results]
    metrics = evaluate_keyword_retrieval(
        retrieved_texts,
        case.get("expected_keywords", []),
    )
    return {
        "id": case_id,
        "question": case["question"],
        "retrieved": results,
        "metrics": metrics,
    }


def evaluate_dataset(
    client: httpx.Client,
    cases: list[dict[str, Any]],
    *,
    top_k: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float]] = []
    for case in cases:
        row = evaluate_case(client, case, top_k=top_k)
        rows.append(row)
        metric_rows.append(row["metrics"])
    return {
        "average": average_metrics(metric_rows),
        "cases": rows,
    }
