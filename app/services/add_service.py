from __future__ import annotations

from app.ingestion.pipeline import IngestionPipeline
from app.schemas import AddRequest, AddResponse


class AddService:
    def __init__(self, *, pipeline: IngestionPipeline) -> None:
        self.pipeline = pipeline

    async def handle(self, request: AddRequest) -> AddResponse:
        return await self.pipeline.run(request)
