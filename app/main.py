from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.api.routes import router
from app.config import Settings, get_settings
from app.core.logging import configure_logging
from app.memory.extractor import build_extractor
from app.retrieval.embedding import build_embedding_provider
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.packer import EvidencePacker
from app.retrieval.reranker import IdentityReranker
from app.services.add_service import AddService
from app.services.container import AppServices
from app.services.search_service import SearchService
from app.storage.sqlite import SQLiteMemoryStore


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    store = SQLiteMemoryStore(settings.database_file)
    embedder = build_embedding_provider(settings)
    extractor = build_extractor()
    retriever = HybridRetriever(
        store=store,
        embedder=embedder,
        candidate_k=settings.retrieval_candidate_k,
    )
    reranker = IdentityReranker()
    packer = EvidencePacker(settings)

    services = AppServices(
        settings=settings,
        store=store,
        embedder=embedder,
        add=AddService(store=store, extractor=extractor, embedder=embedder),
        search=SearchService(
            settings=settings,
            retriever=retriever,
            reranker=reranker,
            packer=packer,
        ),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await store.init()
        app.state.services = services
        yield

    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.include_router(router)

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        print(
            f"{request.method} {request.url.path} "
            f"status={response.status_code} elapsed_ms={elapsed_ms:.2f}"
        )
        return response

    return app


app = create_app()
