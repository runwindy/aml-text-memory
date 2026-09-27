from __future__ import annotations

from typing import Protocol, Sequence

from app.retrieval.hybrid import MemoryHit, tokenize


class Reranker(Protocol):
    async def rerank(self, query: str, hits: Sequence[MemoryHit]) -> list[MemoryHit]:
        ...


class IdentityReranker:
    """Baseline reranker. Replace with a cross-encoder or LLM reranker."""

    async def rerank(self, query: str, hits: Sequence[MemoryHit]) -> list[MemoryHit]:
        return list(hits)


class LexicalReranker:
    """Deterministic reranker using query-term overlap.

    This is a real first-stage reranker for the local baseline and requires no
    external model. Later replace it with bge-reranker or another allowed model.
    """

    async def rerank(self, query: str, hits: Sequence[MemoryHit]) -> list[MemoryHit]:
        query_terms = set(tokenize(query))
        if not query_terms:
            return list(hits)

        reranked: list[MemoryHit] = []
        for hit in hits:
            content_terms = set(tokenize(hit.record.content))
            overlap = len(query_terms & content_terms) / max(len(query_terms), 1)
            new_score = 0.6 * float(hit.score) + 0.4 * overlap
            reranked.append(MemoryHit(record=hit.record, score=new_score))

        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked
