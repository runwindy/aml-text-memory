from __future__ import annotations

import asyncio
import math
from typing import Sequence

from app.config import Settings
from app.decomposer.graph_builder import build_memory_records
from app.decomposer.llm_decomposer import LLMDecomposer
from app.ingestion.models import CanonicalMessage
from app.memory.models import MemoryRecord
from app.memory.multigranularity import build_session_summaries
from app.memory.window_extractor import DialogueWindowExtractor
from app.organizer.llm import LLMOrganizer
from app.organizer.rule_based import RuleBasedOrganizer
from app.schemas import AddRequest


_OPENAI_COMPATIBLE_PROVIDERS = {
    "openai",
    "openai-compatible",
    "openai_compatible",
    "deepseek",
    "deepseek-v4",
    "deepseek-v4.1",
}


def _is_openai_compatible_provider(provider: str) -> bool:
    return provider.strip().lower() in _OPENAI_COMPATIBLE_PROVIDERS


def _is_deepseek_provider(provider: str) -> bool:
    return provider.strip().lower().startswith("deepseek")


def build_organizer(settings: Settings):
    if (
        _is_openai_compatible_provider(settings.organizer_provider)
        and settings.organizer_api_base
        and settings.organizer_api_key
    ):
        return LLMOrganizer(
            api_base=settings.organizer_api_base,
            api_key=settings.organizer_api_key,
            model=settings.organizer_model,
            max_tokens=settings.organizer_max_tokens,
            timeout=settings.organizer_timeout,
            provider=settings.organizer_provider,
            disable_thinking=_is_deepseek_provider(settings.organizer_provider),
        )
    return RuleBasedOrganizer()


def build_decomposer(settings: Settings):
    if (
        _is_openai_compatible_provider(settings.decomposer_provider)
        and settings.decomposer_api_base
        and settings.decomposer_api_key
    ):
        return LLMDecomposer(
            api_base=settings.decomposer_api_base,
            api_key=settings.decomposer_api_key,
            model=settings.decomposer_model,
            max_tokens=settings.decomposer_max_tokens,
            timeout=settings.decomposer_timeout,
            provider=settings.decomposer_provider,
            disable_thinking=_is_deepseek_provider(settings.decomposer_provider),
        )
    return None


def _dedupe_structured(records: list[MemoryRecord]) -> list[MemoryRecord]:
    """Deduplicate only genuinely equivalent structured records.

    Timestamp and source message ids are part of the key.  This keeps repeated
    events / repeated statements at different times as separate Gold records,
    while still collapsing duplicated extractions from overlapping windows.
    """

    by_key: dict[
        tuple[
            str,
            str | None,
            str | None,
            str | None,
            int | None,
            tuple[str, ...],
        ],
        MemoryRecord,
    ] = {}
    for record in records:
        key = (
            record.memory_type,
            record.subject,
            record.predicate,
            record.object_value,
            record.timestamp,
            tuple(record.source_message_ids),
        )
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
        self.multi_granularity_enabled = settings.multi_granularity_enabled
        self.session_summary_max_chars = settings.session_summary_max_chars

    def raw_records(
        self,
        request: AddRequest,
        messages: Sequence[CanonicalMessage],
    ) -> list[MemoryRecord]:
        windows = self.window_extractor.build_windows(request, messages)
        return [window.to_memory_record() for window in windows]

    async def extract(
        self,
        request: AddRequest,
        messages: Sequence[CanonicalMessage],
    ) -> list[MemoryRecord]:
        windows = self.window_extractor.build_windows(request, messages)
        raw_records = [window.to_memory_record() for window in windows]

        structured_records: list[MemoryRecord] = []
        if self.decomposer is not None:
            decomposed = await self._decompose_windows(windows)
            for window in windows:
                result = decomposed.get(window.window_id)
                if result is None:
                    continue
                window_records = build_memory_records(
                    request=request,
                    window=window,
                    result=result,
                )
                for record in window_records:
                    record.metadata["decomposer_provider"] = getattr(
                        self.decomposer, "provider", "unknown"
                    )
                    record.metadata["decomposer_model"] = getattr(
                        self.decomposer, "model", "unknown"
                    )
                structured_records.extend(window_records)
        else:
            for window in windows:
                window_records = await self.organizer.extract(request, window.messages)
                structured_records.extend(window_records)

        summary_records: list[MemoryRecord] = []
        if self.multi_granularity_enabled and messages:
            summary_records = build_session_summaries(
                request=request,
                messages=messages,
                max_chars=self.session_summary_max_chars,
            )

        return raw_records + _dedupe_structured(structured_records) + summary_records

    async def _decompose_windows(self, windows):
        """Split large window sets into bounded concurrent LLM batches."""

        if self.decomposer is None or not windows:
            return {}
        if self.decomposer_concurrency <= 1 or len(windows) == 1:
            return await self.decomposer.decompose_many(windows)

        batch_size = max(1, math.ceil(len(windows) / self.decomposer_concurrency))
        batches = [windows[i : i + batch_size] for i in range(0, len(windows), batch_size)]
        semaphore = asyncio.Semaphore(self.decomposer_concurrency)

        async def run_batch(batch):
            async with semaphore:
                return await self.decomposer.decompose_many(batch)

        parts = await asyncio.gather(*(run_batch(batch) for batch in batches))
        merged = {}
        for part in parts:
            merged.update(part)
        return merged
