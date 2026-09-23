from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

from app.eval.harness import auth_headers, evaluate_dataset, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Add/Search retrieval with a local JSONL dataset.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--key", default=None, help="Memory System Key, sent via X-Api-Key")
    parser.add_argument("--dataset", default="examples/sample_eval.jsonl")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    args = parser.parse_args()

    cases = load_jsonl(args.dataset)
    headers = auth_headers(args.key)

    with httpx.Client(base_url=args.base_url.rstrip("/"), headers=headers, timeout=120.0) as client:
        report = evaluate_dataset(client, cases, top_k=args.top_k)

    if args.output:
        Path(args.output).write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(report["average"], ensure_ascii=False, indent=2))
    for row in report["cases"]:
        metrics = row["metrics"]
        print(
            f"{row['id']}: "
            f"hit@5={metrics.get('hit@5', 0):.3f} "
            f"recall@5={metrics.get('recall@5', 0):.3f} "
            f"mrr={metrics.get('mrr', 0):.3f}"
        )


if __name__ == "__main__":
    main()
