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
    """BGE cross-encoder loaded through transformers, without scikit-learn.

    This avoids sentence-transformers/sklearn DLL issues on Windows while still
    using the GPU and fp16.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.model_name = model_name
        self.batch_size = max(1, batch_size)
        self.max_length = max(64, max_length)

        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        dtype = torch.float16 if device.startswith("cuda") else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=dtype,
            )
        except TypeError:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self._lock = threading.Lock()

    def _predict_scores(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        with self._lock:
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start : start + self.batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                with self._torch.no_grad():
                    logits = self.model(**inputs).logits
                if logits.ndim == 2 and logits.shape[1] == 1:
                    batch_scores = logits.squeeze(-1)
                elif logits.ndim == 2 and logits.shape[1] == 2:
                    batch_scores = logits[:, 1] - logits[:, 0]
                else:
                    batch_scores = logits.reshape(-1)
                scores.extend(batch_scores.float().cpu().tolist())
        return scores

    async def rerank(self, query: str, hits: Sequence[MemoryHit]) -> list[MemoryHit]:
        if not hits:
            return []
        pairs = [(query, hit.record.content) for hit in hits]
        scores = await asyncio.to_thread(self._predict_scores, pairs)
        reranked = [
            MemoryHit(record=hit.record, score=float(score))
            for hit, score in zip(hits, scores)
        ]
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked


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
