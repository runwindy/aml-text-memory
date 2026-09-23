from __future__ import annotations

from datetime import datetime

from app.config import Settings
from app.retrieval.hybrid import MemoryHit
from app.schemas import SearchResponseItem


class EvidencePacker:
    """Convert retrieval hits into the exact Search response contract."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def pack(self, hits: list[MemoryHit], top_k: int) -> list[SearchResponseItem]:
        limit = min(max(top_k, 0), self.settings.max_return_items)
        packed: list[SearchResponseItem] = []
        seen_content: set[str] = set()

        for hit in hits:
            if len(packed) >= limit:
                break

            content = hit.record.content
            if not isinstance(content, str):
                content = str(content)
            content = content.strip()
            if not content or content in seen_content:
                continue

            if self.settings.max_content_chars > 0 and len(content) > self.settings.max_content_chars:
                content = content[: self.settings.max_content_chars].rstrip() + " ..."

            created_at = None
            if hit.record.created_at:
                try:
                    created_at = datetime.fromisoformat(hit.record.created_at)
                except ValueError:
                    created_at = None

            packed.append(
                SearchResponseItem(
                    id=hit.record.id,
                    content=content,
                    score=round(float(hit.score), 6),
                    created_at=created_at,
                )
            )
            seen_content.add(content)

        return packed

