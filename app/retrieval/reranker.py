from __future__ import annotations

import asyncio
import threading
from typing import Protocol, Sequence

from app.retrieval.hybrid import MemoryHit, tokenize


class Reranker(Protocol):
    async def rerank(self, query: str, hits: Sequence[MemoryHit]) -> list[MemoryHit]:
        ...


class IdentityReranker:
    async def rerank(self, query: str, hits: Sequence[MemoryHit]) -> list[MemoryHit]:
        return list(hits)


class LexicalReranker:
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


class CrossEncoderReranker:
    """Local cross-encoder reranker with GPU/VRAM guardrails."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.batch_size = max(1, batch_size)
        self.max_length = max(64, max_length)
        self.model = CrossEncoder(
            model_name,
            device=device,
            max_length=self.max_length,
        )
        self._lock = threading.Lock()

    async def rerank(self, query: str, hits: Sequence[MemoryHit]) -> list[MemoryHit]:
        if not hits:
            return []

        pairs = [(query, hit.record.content) for hit in hits]

        def run() -> list[MemoryHit]:
            # Serialize local cross-encoder calls to avoid concurrent GPU OOM.
            with self._lock:
                scores = self.model.predict(
                    pairs,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                )
            reranked = [
                MemoryHit(record=hit.record, score=float(score))
                for hit, score in zip(hits, scores)
            ]
            reranked.sort(key=lambda item: item.score, reverse=True)
            return reranked

        return await asyncio.to_thread(run)


def build_reranker(settings) -> Reranker:
    provider = (settings.reranker_provider or "lexical").lower()
    if provider == "bge":
        try:
            return CrossEncoderReranker(
                model_name=settings.reranker_model,
                device=settings.reranker_device,
                batch_size=settings.reranker_batch_size,
                max_length=settings.reranker_max_length,
            )
        except Exception:
            return LexicalReranker()
    return LexicalReranker()
