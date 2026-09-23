from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, Sequence

import httpx

from app.config import Settings


class EmbeddingProvider(Protocol):
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class HashingEmbeddingProvider:
    """Offline deterministic baseline embedding.

    Useful for tests and local development. It is not a replacement for a real
    semantic embedding model.
    """

    def __init__(self, dimension: int = 256) -> None:
        if dimension <= 0:
            raise ValueError("embedding dimension must be positive")
        self.dimension = dimension

    def _embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest, "big") % self.dimension
            sign = 1.0 if digest[0] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]


class OpenAICompatibleEmbeddingProvider:
    """OpenAI-compatible /embeddings client.

    Configure this for text-embedding-v4 in the open-source/academic track.
    """

    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        model: str,
        batch_size: int = 64,
        timeout: float = 120.0,
    ) -> None:
        if not api_base:
            raise ValueError("embedding_api_base is required")
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start : start + self.batch_size])
                response = await client.post(
                    f"{self.api_base}/embeddings",
                    headers=headers,
                    json={"model": self.model, "input": batch},
                )
                response.raise_for_status()
                payload = response.json()
                data = payload.get("data", [])
                data = sorted(data, key=lambda item: item.get("index", 0))
                vectors.extend([item["embedding"] for item in data])
        return vectors


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "hashing":
        return HashingEmbeddingProvider(dimension=settings.embedding_dim)
    if settings.embedding_provider == "openai":
        return OpenAICompatibleEmbeddingProvider(
            api_base=settings.embedding_api_base,
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
        )
    raise ValueError(f"unsupported embedding provider: {settings.embedding_provider}")

