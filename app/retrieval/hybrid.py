from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from app.memory.models import MemoryRecord
from app.retrieval.embedding import EmbeddingProvider
from app.retrieval.query_analyzer import QueryPlan
from app.storage.base import MemoryStore

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text or "")]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    dot = sum(left[index] * right[index] for index in range(size))
    left_norm = math.sqrt(sum(value * value for value in left[:size]))
    right_norm = math.sqrt(sum(value * value for value in right[:size]))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def bm25_scores(query: str, documents: Sequence[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Pure-Python BM25 baseline for small candidate sets."""

    query_tokens = tokenize(query)
    tokenized_documents = [tokenize(document) for document in documents]
    if not query_tokens or not tokenized_documents:
        return [0.0] * len(documents)

    document_count = len(tokenized_documents)
    average_length = sum(len(document) for document in tokenized_documents) / max(document_count, 1)
    document_frequency: Counter[str] = Counter()
    for document in tokenized_documents:
        document_frequency.update(set(document))

    scores: list[float] = []
    for document in tokenized_documents:
        frequencies = Counter(document)
        length = len(document)
        score = 0.0
        for token in query_tokens:
            frequency = frequencies.get(token, 0)
            if frequency == 0:
                continue
            df = document_frequency.get(token, 0)
            idf = math.log(1 + (document_count - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (1 - b + b * length / max(average_length, 1e-9))
            score += idf * (frequency * (k1 + 1)) / denominator
        scores.append(score)
    return scores


def reciprocal_rank_fusion(rankings: Iterable[Sequence[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, identifier in enumerate(ranking, start=1):
            scores[identifier] += 1.0 / (k + rank)
    return dict(scores)


@dataclass(slots=True)
class MemoryHit:
    record: MemoryRecord
    score: float


class HybridRetriever:
    """Dense + BM25 fusion baseline.

    This class is the main retrieval extension point. Add graph expansion,
    temporal scoring, query decomposition, and cross-encoder reranking here.
    """

    def __init__(self, store: MemoryStore, embedder: EmbeddingProvider, candidate_k: int = 500) -> None:
        self.store = store
        self.embedder = embedder
        self.candidate_k = candidate_k

    async def retrieve(self, *, user_id: str, plan: QueryPlan, limit: int) -> list[MemoryHit]:
        records = await self.store.fetch_for_user(user_id, self.candidate_k)
        if not records:
            return []

        by_id = {record.id: record for record in records}

        dense_scores: list[tuple[str, float]] = []
        if plan.retrieval_text.strip():
            query_vector = (await self.embedder.embed([plan.retrieval_text]))[0]
            for record in records:
                if not record.embedding:
                    continue
                dense_scores.append((record.id, cosine_similarity(query_vector, record.embedding)))

        sparse_values = bm25_scores(plan.retrieval_text, [record.content for record in records])
        sparse_scores = [
            (record.id, score)
            for record, score in zip(records, sparse_values)
            if score > 0
        ]

        dense_ranking = [
            identifier
            for identifier, _ in sorted(dense_scores, key=lambda item: item[1], reverse=True)
        ]
        sparse_ranking = [
            identifier
            for identifier, _ in sorted(sparse_scores, key=lambda item: item[1], reverse=True)
        ]

        fused = reciprocal_rank_fusion([dense_ranking, sparse_ranking])
        if not fused:
            fused = {record.id: 0.0 for record in records[:limit]}

        max_score = max(fused.values()) if fused else 1.0
        ranked_ids = sorted(fused, key=lambda identifier: fused[identifier], reverse=True)

        hits: list[MemoryHit] = []
        for identifier in ranked_ids[:limit]:
            record = by_id.get(identifier)
            if record is None:
                continue
            normalized = fused[identifier] / max_score if max_score else 0.0
            hits.append(MemoryHit(record=record, score=normalized))
        return hits
