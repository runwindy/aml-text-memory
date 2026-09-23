from __future__ import annotations

from app.memory.extractor import MemoryExtractor
from app.retrieval.embedding import EmbeddingProvider
from app.schemas import AddRequest, AddResponse
from app.storage.base import MemoryStore


class AddService:
    def __init__(
        self,
        *,
        store: MemoryStore,
        extractor: MemoryExtractor,
        embedder: EmbeddingProvider,
    ) -> None:
        self.store = store
        self.extractor = extractor
        self.embedder = embedder

    async def handle(self, request: AddRequest) -> AddResponse:
        existing = await self.store.get_add_response(request.request_id)
        if existing is not None:
            return AddResponse.model_validate(existing)

        records = await self.extractor.extract(request)
        if records:
            embeddings = await self.embedder.embed([record.content for record in records])
            for record, embedding in zip(records, embeddings):
                record.embedding = embedding

        response = await self.store.save_add(request, records)
        return AddResponse.model_validate(response)
