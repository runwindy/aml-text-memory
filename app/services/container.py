from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.retrieval.embedding import EmbeddingProvider
from app.services.add_service import AddService
from app.services.search_service import SearchService
from app.storage.base import MemoryStore


@dataclass(slots=True)
class AppServices:
    settings: Settings
    store: MemoryStore
    embedder: EmbeddingProvider
    add: AddService
    search: SearchService
