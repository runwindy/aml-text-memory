from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import Settings, get_settings
from app.core.errors import RequestConflictError
from app.core.logging import configure_logging
from app.ingestion.pipeline import IngestionPipeline
from app.memory.dialogue_jepa import DialogueJepaPredictor
from app.organizer.composite import CompositeExtractor
from app.retrieval.embedding import build_embedding_provider
from app.retrieval.multi_index import MultiIndexRetriever
from app.retrieval.packer import EvidencePacker
from app.retrieval.reranker import build_reranker
from app.services.add_service import AddService
from app.services.async_worker import AsyncMemoryWorker
from app.services.container import AppServices
from app.services.search_service import SearchService
from app.storage.sqlite import SQLiteMemoryStore


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    store = SQLiteMemoryStore(settings.database_file)
    embedder = build_embedding_provider(settings)
    extractor = CompositeExtractor(settings)
    ingestion = IngestionPipeline(
        store=store,
        extractor=extractor,
        embedder=embedder,
        settings=settings,
    )

    query_expander = None
    if settings.jepa_query_expansion_enabled:
        query_expander = DialogueJepaPredictor.load(
            settings.jepa_model_path,
            device="cpu",
        )
    pseudo_relevance_expansion_enabled = (
        settings.jepa_query_expansion_enabled
        and query_expander is None
        and settings.jepa_query_expansion_fallback
    )

    retriever = MultiIndexRetriever(
        store=store,
        embedder=embedder,
        candidate_k=settings.retrieval_candidate_k,
        dense_k=settings.retrieval_dense_k,
        sparse_k=settings.retrieval_sparse_k,
        graph_max_hops=settings.graph_max_hops,
        graph_beam=settings.graph_beam,
        graph_decay=settings.graph_decay,
        result_window=settings.result_window,
        result_window_seed_k=settings.result_window_seed_k,
        link_prediction_enabled=settings.graph_link_prediction_enabled,
        link_prediction_top_k=settings.graph_link_prediction_top_k,
        link_prediction_threshold=settings.graph_link_prediction_threshold,
        link_prediction_weight=settings.graph_link_prediction_weight,
        link_prediction_max_nodes=settings.graph_link_prediction_max_nodes,
        query_expander=query_expander,
        jepa_blend_weight=settings.jepa_blend_weight,
        pseudo_relevance_expansion_enabled=pseudo_relevance_expansion_enabled,
        pseudo_relevance_seed_k=settings.jepa_pseudo_relevance_seed_k,
        pseudo_relevance_weight=settings.jepa_pseudo_relevance_weight,
        dialogue_tree_expansion_enabled=settings.dialogue_tree_expansion_enabled,
        dialogue_tree_weight=settings.dialogue_tree_weight,
        dialogue_tree_decay=settings.dialogue_tree_decay,
        dialogue_tree_max_hops=settings.dialogue_tree_max_hops,
    )
    reranker = build_reranker(settings)
    packer = EvidencePacker(settings)

    services = AppServices(
        settings=settings,
        store=store,
        embedder=embedder,
        ingestion=ingestion,
        add=AddService(pipeline=ingestion),
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
        worker_task: asyncio.Task | None = None
        if settings.async_worker_enabled:
            worker = AsyncMemoryWorker(
                pipeline=ingestion,
                store=store,
                poll_interval=settings.async_worker_poll_interval,
            )
            worker_task = asyncio.create_task(worker.run())
        try:
            yield
        finally:
            if worker_task is not None:
                worker_task.cancel()
                await asyncio.gather(worker_task, return_exceptions=True)

    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.include_router(router)

    @app.exception_handler(RequestConflictError)
    async def request_conflict_handler(request: Request, exc: RequestConflictError):
        return JSONResponse(
            status_code=409,
            content={"detail": {"reason": str(exc)}},
        )

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





