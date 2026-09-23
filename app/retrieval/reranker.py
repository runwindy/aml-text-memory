from __future__ import annotations

from typing import Protocol, Sequence

from app.retrieval.hybrid import MemoryHit


class Reranker(Protocol):
    async def rerank(self, query: str, hits: Sequence[MemoryHit]) -> list[MemoryHit]:
        ...


class IdentityReranker:
    """Baseline reranker. Replace with a cross-encoder or LLM reranker."""

    async def rerank(self, query: str, hits: Sequence[MemoryHit]) -> list[MemoryHit]:
        return list(hits)
