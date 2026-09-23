from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

_WORD_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def normalize_text(text: object) -> str:
    return " ".join(str(text or "").lower().split())


def tokenize(text: object) -> list[str]:
    return [token.lower() for token in _WORD_RE.findall(str(text or ""))]


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if normalize_text(prediction) == normalize_text(gold) else 0.0


def contains_match(prediction: str, gold: str) -> float:
    gold_norm = normalize_text(gold)
    if not gold_norm:
        return 0.0
    return 1.0 if gold_norm in normalize_text(prediction) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: set[str]) -> float:
    for rank, identifier in enumerate(retrieved_ids, start=1):
        if identifier in relevant_ids:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved = set(retrieved_ids[:k])
    return len(retrieved & relevant_ids) / len(relevant_ids)


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    retrieved = retrieved_ids[:k]
    if not retrieved:
        return 0.0
    return len(set(retrieved) & relevant_ids) / len(retrieved)


def hit_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    return 1.0 if set(retrieved_ids[:k]) & relevant_ids else 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0

    dcg = 0.0
    for rank, identifier in enumerate(retrieved_ids[:k], start=1):
        if identifier in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_id_retrieval(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    *,
    k_values: tuple[int, ...] = (1, 5, 10, 20, 50, 100),
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "mrr": reciprocal_rank(retrieved_ids, relevant_ids),
    }
    for k in k_values:
        metrics[f"hit@{k}"] = hit_at_k(retrieved_ids, relevant_ids, k)
        metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
        metrics[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant_ids, k)
    return metrics


def keyword_in_text(text: str, keyword: str) -> bool:
    keyword_norm = normalize_text(keyword)
    return bool(keyword_norm) and keyword_norm in normalize_text(text)


def evaluate_keyword_retrieval(
    retrieved_texts: Sequence[str],
    expected_keywords: Sequence[str],
    *,
    k_values: tuple[int, ...] = (1, 5, 10, 20, 50, 100),
) -> dict[str, float]:
    """Evaluate retrieval when gold memory IDs are unknown.

    A keyword is counted as found at rank r when any retrieved text at rank <= r
    contains that keyword. This is intentionally permissive and is used as a
    local proxy while benchmark gold labels are unavailable.
    """

    keywords = [keyword for keyword in expected_keywords if normalize_text(keyword)]
    metrics: dict[str, float] = {
        "mrr": 0.0,
        "keyword_total": float(len(keywords)),
    }
    if not keywords:
        for k in k_values:
            metrics[f"hit@{k}"] = 0.0
            metrics[f"recall@{k}"] = 0.0
        return metrics

    first_rank: int | None = None
    for rank, text in enumerate(retrieved_texts, start=1):
        if any(keyword_in_text(text, keyword) for keyword in keywords):
            first_rank = rank
            break
    metrics["mrr"] = 1.0 / first_rank if first_rank else 0.0

    for k in k_values:
        prefix = retrieved_texts[:k]
        found = {keyword for keyword in keywords if any(keyword_in_text(text, keyword) for text in prefix)}
        metrics[f"hit@{k}"] = 1.0 if found else 0.0
        metrics[f"recall@{k}"] = len(found) / len(keywords)
    return metrics


def average_metrics(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = set().union(*(row.keys() for row in rows))
    return {
        key: sum(float(row.get(key, 0.0)) for row in rows) / len(rows)
        for key in sorted(keys)
    }
