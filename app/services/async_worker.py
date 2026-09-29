from __future__ import annotations

import asyncio
import logging

from app.ingestion.pipeline import IngestionPipeline
from app.storage.base import MemoryStore

logger = logging.getLogger(__name__)


class AsyncMemoryWorker:
    """Background worker for raw-ready Add jobs.

    The worker claims jobs from SQLite, builds structured Gold memories and
    graph/entity structures, then marks the job complete.  It is intentionally
    small so it can run in-process for local experiments.
    """

    def __init__(
        self,
        *,
        pipeline: IngestionPipeline,
        store: MemoryStore,
        poll_interval: float = 1.0,
    ) -> None:
        self.pipeline = pipeline
        self.store = store
        self.poll_interval = max(0.1, poll_interval)

    async def run(self) -> None:
        while True:
            job = await self.store.claim_next_async_job()
            if job is None:
                await asyncio.sleep(self.poll_interval)
                continue
            try:
                await self.pipeline.process_async_job(job)
                await self.store.complete_async_job(str(job["job_id"]))
                await self.store.mark_dirty_entities_processed(str(job["user_id"]))
            except Exception as exc:
                logger.warning("async memory job failed: %s", job.get("job_id"), exc_info=True)
                try:
                    await self.store.fail_async_job(str(job["job_id"]), str(exc))
                except Exception:
                    logger.warning("failed to mark async job as failed", exc_info=True)
