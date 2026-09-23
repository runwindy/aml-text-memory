from __future__ import annotations

from app.config import Settings
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.packer import EvidencePacker
from app.retrieval.query_analyzer import analyze_query
from app.retrieval.reranker import Reranker
from app.schemas import SearchRequest, SearchResponse


class SearchService:
    def __init__(
        self,
        *,
        settings: Settings,
        retriever: HybridRetriever,
        reranker: Reranker,
        packer: EvidencePacker,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.reranker = reranker
        self.packer = packer

    async def handle(self, request: SearchRequest) -> SearchResponse:
        plan = analyze_query(request)
        internal_limit = min(max(request.top_k, 1), self.settings.max_return_items)

        hits = await self.retriever.retrieve(
            user_id=request.user_id,
            plan=plan,
            limit=internal_limit,
        )
        hits = await self.reranker.rerank(plan.retrieval_text, hits)
        items = self.packer.pack(hits, request.top_k)
        return SearchResponse(data=items)
