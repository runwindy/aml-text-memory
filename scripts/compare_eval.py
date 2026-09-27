from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_METRICS = (
    "hit@1",
    "hit@5",
    "hit@10",
    "recall@1",
    "recall@5",
    "recall@10",
    "mrr",
    "forbidden@1",
)


def load_metrics(path: str) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    metrics = payload.get("metrics") or payload.get("average")
    if not isinstance(metrics, dict):
        raise SystemExit(f"no metrics found in {path}")
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two local retrieval evaluation reports.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--metrics", nargs="*", default=list(DEFAULT_METRICS))
    args = parser.parse_args()

    baseline = load_metrics(args.baseline)
    current = load_metrics(args.current)

    print(f"{'metric':<16} {'baseline':>10} {'current':>10} {'delta':>10}")
    print("-" * 50)
    for metric in args.metrics:
        old = baseline.get(metric)
        new = current.get(metric)
        if old is None or new is None:
            continue
        delta = new - old
        sign = "+" if delta >= 0 else ""
        print(f"{metric:<16} {old:>10.4f} {new:>10.4f} {sign}{delta:>9.4f}")


if __name__ == "__main__":
    main()
