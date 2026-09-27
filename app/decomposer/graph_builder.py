from __future__ import annotations

from app.decomposer.normalizer import records_from_decomposition
from app.decomposer.schemas import DecompositionResult
from app.ingestion.models import CanonicalMessage
from app.memory.models import MemoryRecord
from app.memory.window_extractor import DialogueWindow
from app.schemas import AddRequest


def build_memory_records(
    *,
    request: AddRequest,
    window: DialogueWindow,
    result: DecompositionResult,
) -> list[MemoryRecord]:
    return records_from_decomposition(
        request=request,
        window=window,
        result=result,
    )
