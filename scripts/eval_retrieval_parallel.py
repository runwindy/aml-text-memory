from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx

from app.eval.harness import add_session, auth_headers, search_case
from app.eval.metrics import average_metrics, evaluate_keyword_retrieval


def evaluate_question(
    *,
    base_url: str,
    headers: dict[str, str],
    user_id: str,
    question: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    with httpx.Client(base_url=base_url, headers=headers, timeout=120.0) as client:
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


def process_group(
    *,
    case: dict[str, Any],
    base_url: str,
    headers: dict[str, str],
    top_k: int,
    workers: int,
) -> list[dict[str, Any]]:
    case_id = str(case["id"])
    user_id = str(case["user_id"])

    sessions = case.get("sessions", [])
    if sessions:
        def add_one(session: dict[str, Any]) -> None:
            with httpx.Client(base_url=base_url, headers=headers, timeout=180.0) as client:
                add_session(client, case_id=case_id, user_id=user_id, session=session)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(add_one, sessions))

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                evaluate_question,
                base_url=base_url,
                headers=headers,
                user_id=user_id,
                question=question,
                top_k=top_k,
            ): question
            for question in case.get("questions", [])
        }
        for future in as_completed(futures):
            row = future.result()
            row["group_id"] = case_id
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel LoCoMo-Refined retrieval evaluator.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--key", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    cases = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    headers = auth_headers(args.key)

    rows: list[dict[str, Any]] = []
    for case in cases:
        group_rows = process_group(
            case=case,
            base_url=args.base_url.rstrip("/"),
            headers=headers,
            top_k=args.top_k,
            workers=args.workers,
        )
        rows.extend(group_rows)
        print(
            f"group={case['id']} questions={len(group_rows)} total={len(rows)}",
            flush=True,
        )

    report = {
        "average": average_metrics([row["metrics"] for row in rows]),
        "cases": rows,
    }
    output_path = Path(args.output)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report["average"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
