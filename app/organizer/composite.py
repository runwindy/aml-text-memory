from __future__ import annotations

import asyncio

from typing import Sequence

from app.config import Settings
from app.decomposer.graph_builder import build_memory_records
from app.decomposer.llm_decomposer import LLMDecomposer
from app.ingestion.models import CanonicalMessage
from app.memory.models import MemoryRecord
from app.memory.window_extractor import DialogueWindowExtractor
from app.organizer.llm import LLMOrganizer
from app.organizer.rule_based import RuleBasedOrganizer
from app.schemas import AddRequest


def build_organizer(settings: Settings):
    if (
        settings.organizer_provider == "openai"
        and settings.organizer_api_base
        and settings.organizer_api_key
    ):
        return LLMOrganizer(
            api_base=settings.organizer_api_base,
            api_key=settings.organizer_api_key,
            model=settings.organizer_model,
            max_tokens=settings.organizer_max_tokens,
            timeout=settings.organizer_timeout,
        )
    return RuleBasedOrganizer()


def build_decomposer(settings: Settings):
    if (
        settings.decomposer_provider == "openai"
        and settings.decomposer_api_base
        and settings.decomposer_api_key
    ):
        return LLMDecomposer(
            api_base=settings.decomposer_api_base,
            api_key=settings.decomposer_api_key,
            model=settings.decomposer_model,
            max_tokens=settings.decomposer_max_tokens,
            timeout=settings.decomposer_timeout,
        )
    return None


def _dedupe_structured(records: list[MemoryRecord]) -> list[MemoryRecord]:
    by_key: dict[tuple[str, str | None, str | None, str | None], MemoryRecord] = {}
    for record in records:
        key = (record.memory_type, record.subject, record.predicate, record.object_value)
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = record
            continue

        existing.source_message_ids = list(
            dict.fromkeys(existing.source_message_ids + record.source_message_ids)
        )
        if (record.confidence or 0.0) > (existing.confidence or 0.0):
            existing.confidence = record.confidence
            existing.content = record.content
            existing.metadata.update(record.metadata)

    return list(by_key.values())


class CompositeExtractor:
    """Dialogue-window raw evidence + decomposer / organizer structured Gold."""

    def __init__(self, settings: Settings) -> None:
        self.window_extractor = DialogueWindowExtractor(
            window_size=settings.window_size,
            overlap=settings.window_overlap,
        )
        self.organizer = build_organizer(settings)
        self.decomposer = build_decomposer(settings)
        self.decomposer_concurrency = max(1, settings.decomposer_concurrency)

    async def extract(
        self,
        request: AddRequest,
        messages: Sequence[CanonicalMessage],
    ) -> list[MemoryRecord]:
        windows = self.window_extractor.build_windows(request, messages)
        raw_records = [window.to_memory_record() for window in windows]

        structured_records: list[MemoryRecord] = []
        if self.decomposer is not None:
            decomposed = await self.decomposer.decompose_many(windows)
            for window in windows:
                result = decomposed.get(window.window_id)
                if result is None:
                    continue
                structured_records.extend(
                    build_memory_records(
                        request=request,
                        window=window,
                        result=result,
                    )
                )
        else:
            for window in windows:
                window_records = await self.organizer.extract(request, window.messages)
                structured_records.extend(window_records)

        return raw_records + _dedupe_structured(structured_records)
