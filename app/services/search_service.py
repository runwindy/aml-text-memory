from __future__ import annotations

from app.config import Settings
from app.retrieval.multi_index import MultiIndexRetriever
from app.retrieval.packer import EvidencePacker
from app.retrieval.query_analyzer import analyze_query
from app.memory.predictive import apply_predictive_bias
from app.retrieval.reranker import Reranker
from app.schemas import SearchRequest, SearchResponse


class SearchService:
    def __init__(
        self,
        *,
        settings: Settings,
        retriever: MultiIndexRetriever,
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

        rerank_limit = self.settings.reranker_candidates
        if rerank_limit and rerank_limit > 0 and len(hits) > rerank_limit:
            # Cross-encoder reranking is expensive.  Rerank only the strongest
            # fused candidates and preserve the remaining order as a tail.
            head = await self.reranker.rerank(plan.retrieval_text, hits[:rerank_limit])
            hits = head + hits[rerank_limit:]
        else:
            hits = await self.reranker.rerank(plan.retrieval_text, hits)

        if self.settings.predictive_memory_enabled:
            hits = apply_predictive_bias(
                hits,
                weight=self.settings.predictive_rerank_weight,
            )

        items = self.packer.pack(hits, request.top_k)
        return SearchResponse(data=items)

