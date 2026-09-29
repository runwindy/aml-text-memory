from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from app.config import Settings
from app.core.errors import RequestConflictError
from app.ingestion.deduplicator import deduplicate
from app.ingestion.models import RawAddRequest
from app.ingestion.normalizer import normalize_messages
from app.ingestion.safety import process as process_safety
from app.ingestion.time_processor import process as process_time
from app.ingestion.validator import validate_request
from app.memory.extractor import MemoryExtractor
from app.memory.predictive import annotate_prediction_surprise
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
        settings: Settings | None = None,
    ) -> None:
        self.store = store
        self.extractor = extractor
        self.embedder = embedder
        self.settings = settings

    async def run(self, request: AddRequest) -> AddResponse:
        validate_request(request)
        request_hash = payload_hash(request)

        now = datetime.now(timezone.utc).isoformat()
        raw = RawAddRequest(
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
            payload_json=request.model_dump_json(),
            payload_hash=request_hash,
            received_at=now,
        )

        # Atomically claim the request id before doing expensive work.  A second
        # concurrent writer cannot create the same logical request, even if both
        # processes passed the initial "not found" check.
        existing = await self.store.claim_add(
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
            payload_hash=request_hash,
        )
        if existing is not None:
            if (
                existing.get("user_id") != request.user_id
                or existing.get("session_id") != request.session_id
                or (
                    existing.get("payload_hash") is not None
                    and existing["payload_hash"] != request_hash
                )
            ):
                raise RequestConflictError(
                    "request_id was already used with a different payload"
                )

            status = existing.get("status")
            if status == "succeeded":
                cached = existing.get("response_json")
                if isinstance(cached, str):
                    cached = json.loads(cached)
                return AddResponse.model_validate(cached)

            if status == "failed":
                # The previous attempt failed before publishing Silver/Gold.
                # Bronze is already durable, so retrying is safe and necessary.
                await self.store.reset_add_for_retry(request.request_id)
            else:
                # Another worker is processing the same request.  Wait briefly
                # for it to finish instead of returning a false success.
                cached = await self.store.wait_for_add_response(
                    request.request_id,
                    timeout=60.0,
                )
                if cached is not None:
                    return AddResponse.model_validate(cached)
                raise RequestConflictError(
                    "request_id is still being processed; retry later"
                )

        try:
            # Persist Bronze *before* normalization, extraction, embedding, or
            # any external model call.  If the downstream pipeline fails, the
            # original request remains durable and can be retried.
            await self.store.save_raw_request(raw, status="pending")

            messages = normalize_messages(request)
            messages = deduplicate(messages)
            messages = process_time(messages)
            messages = process_safety(messages)

            # Keep exact duplicates in Bronze/Silver.  They may encode repeated
            # events or emphasis, so Gold extraction must see every occurrence.
            primary_messages = [
                message for message in messages if message.normalized_content
            ]

            if self.settings is not None and self.settings.async_processing_enabled:
                raw_records = self.extractor.raw_records(request, primary_messages)
                response = await self.store.save_raw_ready_result(
                    request=request,
                    messages=messages,
                    records=raw_records,
                    payload_hash=request_hash,
                )
                return AddResponse.model_validate(response)

            memories = await self.extractor.extract(request, primary_messages)

            if (
                memories
                and self.settings is not None
                and self.settings.predictive_memory_enabled
            ):
                # JEPA-inspired assistant-consistency signal.  This is
                # auxiliary metadata; the main embedding pass still follows.
                await annotate_prediction_surprise(
                    memories,
                    self.embedder,
                    surprise_threshold=self.settings.prediction_surprise_threshold,
                )

            if memories:
                vectors = await self.embedder.embed([memory.content for memory in memories])
                for memory, vector in zip(memories, vectors):
                    memory.embedding = vector

            response = await self.store.save_silver_gold(
                request=request,
                messages=messages,
                records=memories,
                payload_hash=request_hash,
            )
            return AddResponse.model_validate(response)
        except Exception as exc:
            # Keep Bronze and mark the job retryable.  Never return success until
            # Silver/Gold are committed and the records are immediately searchable.
            try:
                await self.store.mark_add_failed(request.request_id, str(exc))
            except Exception:
                # Preserve the original pipeline error for the API caller.
                pass
            raise

    async def process_async_job(self, job: dict) -> dict:
        """Run the structured part of an Add request after raw ingestion."""

        request = AddRequest.model_validate_json(str(job["payload_json"]))
        validate_request(request)
        messages = normalize_messages(request)
        messages = deduplicate(messages)
        messages = process_time(messages)
        messages = process_safety(messages)
        primary_messages = [message for message in messages if message.normalized_content]

        memories = await self.extractor.extract(request, primary_messages)
        if (
            memories
            and self.settings is not None
            and self.settings.predictive_memory_enabled
        ):
            await annotate_prediction_surprise(
                memories,
                self.embedder,
                surprise_threshold=self.settings.prediction_surprise_threshold,
            )
        if memories:
            vectors = await self.embedder.embed([memory.content for memory in memories])
            for memory, vector in zip(memories, vectors):
                memory.embedding = vector

        return await self.store.save_silver_gold(
            request=request,
            messages=messages,
            records=memories,
            payload_hash=payload_hash(request),
        )
