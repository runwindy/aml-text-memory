from __future__ import annotations

from collections import defaultdict

from app.indexing.entity_index import EntityIndex
from app.indexing.time_index import TimeIndex
from app.indexing.type_index import type_scores
from app.memory.dialogue_jepa import DialogueJepaPredictor, pseudo_relevance_query_vector
from app.memory.models import MemoryRecord
from app.retrieval.conflict import apply_current_state_bias
from app.retrieval.embedding import EmbeddingProvider
from app.retrieval.dialogue_graph import build_tree_graph_scores
from app.retrieval.graph_expander import expand_graph
from app.retrieval.entity_graph import entity_graph_scores
from app.retrieval.graph_jepa import build_predicted_adjacency
from app.retrieval.granularity import granularity_scores
from app.retrieval.hybrid import MemoryHit, bm25_scores, cosine_similarity
from app.retrieval.query_analyzer import QueryPlan
from app.retrieval.query_keywords import extract_query_keywords
from app.retrieval.result_window import expand_result_window
from app.retrieval.structured_matcher import structured_scores
from app.retrieval.temporal_resolver import temporal_scores
from app.storage.base import MemoryStore


class MultiIndexRetriever:
    """Dense + BM25 + entity + time + type + structured + graph retrieval."""

    def __init__(
        self,
        store: MemoryStore,
        embedder: EmbeddingProvider,
        candidate_k: int = 500,
        dense_k: int | None = None,
        sparse_k: int | None = None,
        rrf_k: int = 60,
        graph_max_hops: int = 2,
        graph_beam: int = 10,
        graph_decay: float = 0.8,
        result_window: int = 1,
        result_window_seed_k: int = 20,
        link_prediction_enabled: bool = False,
        link_prediction_top_k: int = 8,
        link_prediction_threshold: float = 0.55,
        link_prediction_weight: float = 0.5,
        link_prediction_max_nodes: int = 200,
        query_expander: DialogueJepaPredictor | None = None,
        jepa_blend_weight: float = 0.5,
        pseudo_relevance_expansion_enabled: bool = False,
        pseudo_relevance_seed_k: int = 20,
        pseudo_relevance_weight: float = 0.5,
        dialogue_tree_expansion_enabled: bool = False,
        dialogue_tree_weight: float = 0.7,
        dialogue_tree_decay: float = 0.7,
        dialogue_tree_max_hops: int = 2,
        multi_granularity_enabled: bool = True,
        granularity_atomic_weight: float = 1.0,
        granularity_window_weight: float = 0.7,
        granularity_session_weight: float = 0.6,
        granularity_multi_session_weight: float = 0.9,
        entity_graph_expansion_enabled: bool = False,
        entity_graph_max_hops: int = 1,
        entity_graph_weight: float = 0.6,
        entity_graph_seed_k: int = 10,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.candidate_k = candidate_k
        self.dense_k = dense_k
        self.sparse_k = sparse_k
        self.rrf_k = rrf_k
        self.graph_max_hops = graph_max_hops
        self.graph_beam = graph_beam
        self.graph_decay = graph_decay
        self.result_window = result_window
        self.result_window_seed_k = result_window_seed_k
        self.link_prediction_enabled = link_prediction_enabled
        self.link_prediction_top_k = link_prediction_top_k
        self.link_prediction_threshold = link_prediction_threshold
        self.link_prediction_weight = link_prediction_weight
        self.link_prediction_max_nodes = link_prediction_max_nodes
        self.query_expander = query_expander
        self.jepa_blend_weight = jepa_blend_weight
        self.pseudo_relevance_expansion_enabled = pseudo_relevance_expansion_enabled
        self.pseudo_relevance_seed_k = pseudo_relevance_seed_k
        self.pseudo_relevance_weight = pseudo_relevance_weight
        self.dialogue_tree_expansion_enabled = dialogue_tree_expansion_enabled
        self.dialogue_tree_weight = dialogue_tree_weight
        self.dialogue_tree_decay = dialogue_tree_decay
        self.dialogue_tree_max_hops = dialogue_tree_max_hops
        self.multi_granularity_enabled = multi_granularity_enabled
        self.granularity_atomic_weight = granularity_atomic_weight
        self.granularity_window_weight = granularity_window_weight
        self.granularity_session_weight = granularity_session_weight
        self.granularity_multi_session_weight = granularity_multi_session_weight
        self.entity_graph_expansion_enabled = entity_graph_expansion_enabled
        self.entity_graph_max_hops = entity_graph_max_hops
        self.entity_graph_weight = entity_graph_weight
        self.entity_graph_seed_k = entity_graph_seed_k

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

        dialogue_nodes: list[dict] = []
        memory_edges: list[dict] = []
        entity_nodes: list[dict] = []
        entity_edges: list[dict] = []
        memory_entity_links: list[dict] = []
        if self.dialogue_tree_expansion_enabled:
            dialogue_nodes = await self.store.fetch_dialogue_nodes(user_id)
            memory_edges = await self.store.fetch_memory_edges(user_id)
        if self.entity_graph_expansion_enabled:
            entity_nodes = await self.store.fetch_entity_nodes(user_id)
            entity_edges = await self.store.fetch_entity_edges(user_id)
            memory_entity_links = await self.store.fetch_memory_entity_links(user_id)

        by_id = {record.id: record for record in records}
        fused: dict[str, float] = defaultdict(float)

        def add_ranking(
            scores: dict[str, float],
            weight: float,
            top_k: int | None = None,
        ) -> None:
            ranking = sorted(scores, key=lambda identifier: scores[identifier], reverse=True)
            if top_k is not None and top_k > 0:
                ranking = ranking[:top_k]
            for rank, identifier in enumerate(ranking):
                fused[identifier] += weight / (self.rrf_k + rank + 1)

        # Dense retrieval.
        dense_scores: dict[str, float] = {}
        query_vector: list[float] | None = None
        if plan.retrieval_text.strip():
            query_vector = (await self.embedder.embed([plan.retrieval_text]))[0]
            if self.query_expander is not None:
                query_vector = self.query_expander.expand_query(
                    query_vector,
                    blend_weight=self.jepa_blend_weight,
                )
            for record in records:
                if record.embedding:
                    dense_scores[record.id] = cosine_similarity(query_vector, record.embedding)
        add_ranking(dense_scores, 1.0, self.dense_k)

        # BM25 retrieval.
        sparse_values = bm25_scores(plan.retrieval_text, [record.content for record in records])
        sparse_scores = {
            record.id: float(score)
            for record, score in zip(records, sparse_values)
            if score > 0
        }
        add_ranking(sparse_scores, 1.0, self.sparse_k)

        # Entity retrieval.
        entity_scores = EntityIndex(records).scores(plan.retrieval_text)
        add_ranking(entity_scores, 0.7)

        # Time retrieval.
        time_index_scores = TimeIndex(records).scores(plan.retrieval_text)
        add_ranking(time_index_scores, 0.4)

        # Memory-type retrieval.
        kind_scores = type_scores(records, plan.query_text)
        add_ranking(kind_scores, 0.5)

        # Multi-granularity bias: atomic nodes are preferred for direct
        # questions; session summaries are preferred for broad questions.
        if self.multi_granularity_enabled:
            grain_scores = granularity_scores(
                records,
                plan,
                atomic_weight=self.granularity_atomic_weight,
                window_weight=self.granularity_window_weight,
                session_weight=self.granularity_session_weight,
                multi_session_weight=self.granularity_multi_session_weight,
            )
            add_ranking(grain_scores, 0.5)

        # Explicit temporal resolver.
        temporal_resolver_scores = temporal_scores(records, plan.query_text)
        add_ranking(temporal_resolver_scores, 0.6)

        # Persistent entity-graph retrieval.
        if self.entity_graph_expansion_enabled and entity_nodes:
            entity_scores = entity_graph_scores(
                records=records,
                query_text=plan.query_text,
                entity_nodes=entity_nodes,
                entity_edges=entity_edges,
                memory_entity_links=memory_entity_links,
                max_hops=self.entity_graph_max_hops,
                seed_k=self.entity_graph_seed_k,
            )
            add_ranking(entity_scores, self.entity_graph_weight)

        # Keyword-driven structured retrieval.
        keywords = extract_query_keywords(plan.query_text + "\n" + plan.retrieval_text)
        structured = structured_scores(records, keywords)
        add_ranking(structured, 0.9)

        # Training-free JEPA-style pseudo-relevance expansion.  The centroid
        # of the strongest current hits is used as a predicted evidence latent
        # and added as an extra dense branch.
        if (
            self.pseudo_relevance_expansion_enabled
            and query_vector is not None
            and fused
        ):
            top_seed_ids = sorted(
                fused,
                key=lambda identifier: fused[identifier],
                reverse=True,
            )[: self.pseudo_relevance_seed_k]
            seed_vectors = [
                by_id[identifier].embedding
                for identifier in top_seed_ids
                if identifier in by_id and by_id[identifier].embedding
            ]
            if seed_vectors:
                expanded_query = pseudo_relevance_query_vector(
                    query_vector,
                    seed_vectors,
                    weight=self.pseudo_relevance_weight,
                )
                pseudo_scores = {
                    record.id: cosine_similarity(expanded_query, record.embedding)
                    for record in records
                    if record.embedding
                }
                add_ranking(pseudo_scores, 0.8, self.dense_k)

        # Graph expansion from the current best seeds.
        seed_ids = sorted(fused, key=lambda identifier: fused[identifier], reverse=True)[:10]
        predicted_edges = None
        if self.link_prediction_enabled:
            predicted_edges = build_predicted_adjacency(
                records,
                query_vector=query_vector,
                top_k=self.link_prediction_top_k,
                threshold=self.link_prediction_threshold,
                weight=self.link_prediction_weight,
                max_nodes=self.link_prediction_max_nodes,
            )

        graph_scores = expand_graph(
            records,
            seed_ids,
            keywords,
            max_hops=self.graph_max_hops,
            beam=self.graph_beam,
            decay=self.graph_decay,
            predicted_edges=predicted_edges,
        )
        add_ranking(graph_scores, 0.8)

        if self.dialogue_tree_expansion_enabled and dialogue_nodes:
            tree_seed_ids = sorted(
                fused,
                key=lambda identifier: fused[identifier],
                reverse=True,
            )[:20]
            seed_scores = {identifier: fused[identifier] for identifier in tree_seed_ids}
            tree_scores = build_tree_graph_scores(
                records=records,
                dialogue_nodes=dialogue_nodes,
                memory_edges=memory_edges,
                seed_scores=seed_scores,
                decay=self.dialogue_tree_decay,
                max_hops=self.dialogue_tree_max_hops,
            )
            add_ranking(tree_scores, self.dialogue_tree_weight)

        if self.result_window:
            seed_ids_for_window = sorted(
                fused,
                key=lambda identifier: fused[identifier],
                reverse=True,
            )[: self.result_window_seed_k]
            fused = expand_result_window(
                records,
                dict(fused),
                seed_ids_for_window,
                window=self.result_window,
                seed_k=self.result_window_seed_k,
            )

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
