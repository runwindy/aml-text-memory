from __future__ import annotations

from collections import defaultdict

from app.indexing.entity_index import EntityIndex
from app.indexing.time_index import TimeIndex
from app.indexing.type_index import type_scores
from app.memory.models import MemoryRecord
from app.retrieval.conflict import apply_current_state_bias
from app.retrieval.embedding import EmbeddingProvider
from app.retrieval.graph_expander import expand_graph
from app.retrieval.hybrid import MemoryHit, bm25_scores, cosine_similarity
from app.retrieval.query_analyzer import QueryPlan
from app.retrieval.query_keywords import extract_query_keywords
from app.retrieval.structured_matcher import structured_scores
from app.storage.base import MemoryStore


class MultiIndexRetriever:
    """Dense + BM25 + entity + time + type + structured + graph retrieval."""

    def __init__(
        self,
        store: MemoryStore,
        embedder: EmbeddingProvider,
        candidate_k: int = 500,
        rrf_k: int = 60,
        graph_max_hops: int = 2,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.graph_max_hops = graph_max_hops

    async def retrieve(
        self,
        *,
        user_id: str,
        plan: QueryPlan,
        limit: int,
    ) -> list[MemoryHit]:
        records = await self.store.fetch_for_user(user_id, self.candidate_k)
        if not records:
            return []

        by_id = {record.id: record for record in records}
        fused: dict[str, float] = defaultdict(float)

        def add_ranking(scores: dict[str, float], weight: float) -> None:
            ranking = sorted(scores, key=lambda identifier: scores[identifier], reverse=True)
            for rank, identifier in enumerate(ranking):
                fused[identifier] += weight / (self.rrf_k + rank + 1)

        # Dense retrieval.
        dense_scores: dict[str, float] = {}
        if plan.retrieval_text.strip():
            query_vector = (await self.embedder.embed([plan.retrieval_text]))[0]
            for record in records:
                if record.embedding:
                    dense_scores[record.id] = cosine_similarity(query_vector, record.embedding)
        add_ranking(dense_scores, 1.0)

        # BM25 retrieval.
        sparse_values = bm25_scores(plan.retrieval_text, [record.content for record in records])
        sparse_scores = {
            record.id: float(score)
            for record, score in zip(records, sparse_values)
            if score > 0
        }
        add_ranking(sparse_scores, 1.0)

        # Entity retrieval.
        entity_scores = EntityIndex(records).scores(plan.retrieval_text)
        add_ranking(entity_scores, 0.7)

        # Time retrieval.
        temporal_scores = TimeIndex(records).scores(plan.retrieval_text)
        add_ranking(temporal_scores, 0.4)

        # Memory-type retrieval.
        kind_scores = type_scores(records, plan.query_text)
        add_ranking(kind_scores, 0.5)

        # Keyword-driven structured retrieval.
        keywords = extract_query_keywords(plan.query_text + "\n" + plan.retrieval_text)
        structured = structured_scores(records, keywords)
        add_ranking(structured, 0.9)

        # Graph expansion from the current best seeds.
        seed_ids = sorted(fused, key=lambda identifier: fused[identifier], reverse=True)[:10]
        graph_scores = expand_graph(
            records,
            seed_ids,
            keywords,
            max_hops=self.graph_max_hops,
        )
        add_ranking(graph_scores, 0.8)

        if not fused:
            fused = {record.id: 0.0 for record in records[:limit]}

        fused = apply_current_state_bias(dict(fused), by_id, plan.query_text)
        max_score = max(fused.values()) if fused else 1.0
        ranked_ids = sorted(fused, key=lambda identifier: fused[identifier], reverse=True)

        hits: list[MemoryHit] = []
        for identifier in ranked_ids[:limit]:
            record = by_id.get(identifier)
            if record is None:
                continue
            normalized = fused[identifier] / max_score if max_score else 0.0
            hits.append(MemoryHit(record=record, score=normalized))
        return hits
