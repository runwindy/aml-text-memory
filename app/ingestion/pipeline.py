from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from app.core.errors import RequestConflictError
from app.ingestion.deduplicator import deduplicate
from app.ingestion.models import RawAddRequest
from app.ingestion.normalizer import normalize_messages
from app.ingestion.safety import process as process_safety
from app.ingestion.time_processor import process as process_time
from app.ingestion.validator import validate_request
from app.memory.extractor import MemoryExtractor
from app.retrieval.embedding import EmbeddingProvider
from app.schemas import AddRequest, AddResponse
from app.storage.base import MemoryStore


def payload_hash(request: AddRequest) -> str:
    payload = request.model_dump(mode="json")
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class IngestionPipeline:
    """Contract-driven ingestion pipeline.

    Bronze raw -> Normalizer -> Deduplicator -> TimeProcessor -> SafetyProcessor
    -> Silver messages -> Extractor -> Gold memories -> Embedding -> Index.
    """

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

    async def run(self, request: AddRequest) -> AddResponse:
        validate_request(request)
        request_hash = payload_hash(request)

        existing = await self.store.get_add_response(request.request_id)
        if existing is not None:
            metadata = await self.store.get_add_metadata(request.request_id)
            if (
                metadata is not None
                and metadata.get("payload_hash") is not None
                and metadata["payload_hash"] != request_hash
            ):
                raise RequestConflictError(
                    "request_id was already used with a different payload"
                )
            return AddResponse.model_validate(existing)

        now = datetime.now(timezone.utc).isoformat()
        raw = RawAddRequest(
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
            payload_json=request.model_dump_json(),
            payload_hash=request_hash,
            received_at=now,
        )

        # Bronze raw is persisted together with Silver/Gold in one transaction.
        messages = normalize_messages(request)
        messages = deduplicate(messages)
        messages = process_time(messages)
        messages = process_safety(messages)

        primary_messages = [
            message for message in messages if "exact_duplicate" not in message.quality_flags
        ]

        memories = await self.extractor.extract(request, primary_messages)
        if memories:
            vectors = await self.embedder.embed([memory.content for memory in memories])
            for memory, vector in zip(memories, vectors):
                memory.embedding = vector

        response = await self.store.save_ingestion_result(
            request=request,
            raw=raw,
            messages=messages,
            records=memories,
        )
        return AddResponse.model_validate(response)
