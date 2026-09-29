from __future__ import annotations

from typing import Protocol, Sequence

from app.ingestion.models import CanonicalMessage, RawAddRequest
from app.memory.models import MemoryRecord
from app.schemas import AddRequest


class MemoryStore(Protocol):
    async def init(self) -> None:
        ...

    async def claim_add(
        self,
        *,
        request_id: str,
        user_id: str,
        session_id: str,
        payload_hash: str,
    ) -> dict | None:
        ...

    async def get_add_response(self, request_id: str) -> dict | None:
        ...

    async def get_add_metadata(self, request_id: str) -> dict | None:
        ...

    async def get_add_record(self, request_id: str) -> dict | None:
        ...

    async def wait_for_add_response(
        self,
        request_id: str,
        *,
        timeout: float = 60.0,
        interval: float = 0.1,
    ) -> dict | None:
        ...

    async def reset_add_for_retry(self, request_id: str) -> None:
        ...

    async def mark_add_failed(self, request_id: str, error_message: str) -> None:
        ...

    async def save_raw_request(self, raw: RawAddRequest, *, status: str = "pending") -> None:
        ...

    async def save_add(self, request: AddRequest, records: Sequence[MemoryRecord]) -> dict:
        ...

    async def save_ingestion_result(
        self,
        *,
        request: AddRequest,
        raw: RawAddRequest,
        messages: Sequence[CanonicalMessage],
        records: Sequence[MemoryRecord],
    ) -> dict:
        ...

    async def save_silver_gold(
        self,
        *,
        request: AddRequest,
        messages: Sequence[CanonicalMessage],
        records: Sequence[MemoryRecord],
        payload_hash: str,
    ) -> dict:
        ...

    async def save_raw_ready_result(
        self,
        *,
        request: AddRequest,
        messages: Sequence[CanonicalMessage],
        records: Sequence[MemoryRecord],
        payload_hash: str,
    ) -> dict:
        ...

    async def claim_next_async_job(self) -> dict | None:
        ...

    async def complete_async_job(self, job_id: str) -> None:
        ...

    async def fail_async_job(self, job_id: str, error_message: str) -> None:
        ...

    async def mark_dirty_entities_processed(self, user_id: str) -> None:
        ...

    async def fetch_for_user(self, user_id: str, limit: int | None = None) -> list[MemoryRecord]:
        ...

    async def fetch_dialogue_nodes(self, user_id: str) -> list[dict]:
        ...

    async def fetch_memory_edges(self, user_id: str) -> list[dict]:
        ...
