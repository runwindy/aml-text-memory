from __future__ import annotations

from typing import Protocol, Sequence

from app.ingestion.models import CanonicalMessage, RawAddRequest
from app.memory.models import MemoryRecord
from app.schemas import AddRequest


class MemoryStore(Protocol):
    async def init(self) -> None:
        ...

    async def get_add_response(self, request_id: str) -> dict | None:
        ...

    async def get_add_metadata(self, request_id: str) -> dict | None:
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

    async def fetch_for_user(self, user_id: str, limit: int | None = None) -> list[MemoryRecord]:
        ...
