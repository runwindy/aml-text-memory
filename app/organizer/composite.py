from __future__ import annotations

from typing import Sequence

from app.ingestion.models import CanonicalMessage
from app.memory.extractor import PassThroughExtractor
from app.memory.models import MemoryRecord
from app.organizer.rule_based import RuleBasedOrganizer
from app.schemas import AddRequest


class CompositeExtractor:
    """Raw evidence extractor + structured Gold organizer."""

    def __init__(self) -> None:
        self.raw = PassThroughExtractor()
        self.organizer = RuleBasedOrganizer()

    async def extract(
        self,
        request: AddRequest,
        messages: Sequence[CanonicalMessage],
    ) -> list[MemoryRecord]:
        raw_records = await self.raw.extract(request, messages)
        structured_records = await self.organizer.extract(request, messages)
        return raw_records + structured_records
