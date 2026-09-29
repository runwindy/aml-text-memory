import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration.

    All values can be provided with an AML_ prefix, for example:
    AML_AUTH_MODE=bearer
    AML_EMBEDDING_PROVIDER=openai
    AML_EMBEDDING_API_BASE=https://api.example.com/v1
    """

    model_config = SettingsConfigDict(
        env_prefix="AML_",
        # Default to .env, but allow an explicit local-experiment file, e.g.
        # AML_ENV_FILE=D:\workspace\memory\aml-text-memory\.env.deepseek
        env_file=os.environ.get("AML_ENV_FILE", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "AML Text Memory"
    environment: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # SQLite is the baseline persistence layer. Replace it with Postgres or a
    # vector database for the full scale evaluation.
    database_path: str = "./data/aml.db"

    # Official Add/Search authentication supports Token, Bearer, X-Api-Key, and
    # none for public smoke. Use "none" only for local development.
    auth_mode: str = "none"
    memory_system_key: str = "change-me"

    # Baseline embedding provider is deterministic and offline. Switch to an
    # OpenAI-compatible endpoint for text-embedding-v4.
    embedding_provider: str = "hashing"
    embedding_model: str = "text-embedding-v4"
    embedding_api_base: str = ""
    embedding_api_key: str = ""
    embedding_dim: int = 256
    embedding_batch_size: int = 64

    # Optional LLM memory organizer. Use gpt-4o-mini for open-source/academic
    # compliance, or leave as "rule" for the deterministic local baseline.
    organizer_provider: str = "rule"
    organizer_model: str = "gpt-4o-mini"
    organizer_api_base: str = ""
    organizer_api_key: str = ""
    organizer_max_tokens: int = 1024
    organizer_timeout: float = 120.0

    # Dialogue-window extraction.
    window_size: int = 3
    window_overlap: int = 1

    # Reranker.
    reranker_provider: str = "lexical"  # lexical | bge
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cuda"
    reranker_batch_size: int = 8
    reranker_max_length: int = 512
    reranker_candidates: int = 50

    # Result window expansion.
    result_window: int = 1
    result_window_seed_k: int = 20

    # Temporal and conflict policy.
    temporal_enrichment: bool = False
    conflict_mode: str = "versioned"  # basic | versioned

    # Graph expansion.
    graph_max_hops: int = 2
    graph_beam: int = 10
    graph_decay: float = 0.8

    # Persistent dialogue-tree / graph expansion during Search.
    dialogue_tree_expansion_enabled: bool = False
    dialogue_tree_weight: float = 0.7
    dialogue_tree_decay: float = 0.7
    dialogue_tree_max_hops: int = 2

    # Optional complex-sentence decomposer.
    # off: use existing Organizer; openai: use decomposition prompt.
    decomposer_provider: str = "off"
    decomposer_model: str = "gpt-4o-mini"
    decomposer_api_base: str = ""
    decomposer_api_key: str = ""
    decomposer_max_tokens: int = 4096
    decomposer_timeout: float = 120.0
    decomposer_concurrency: int = 4

    # JEPA-inspired predictive memory layer.
    #
    # This is an auxiliary signal, not a replacement for text-embedding-v4.
    # When enabled, assistant replies are treated as observed future targets;
    # the distance between the predicted context latent and the actual
    # assistant latent becomes a surprise / update signal.
    predictive_memory_enabled: bool = False
    prediction_surprise_threshold: float = 0.35
    predictive_rerank_weight: float = 0.15

    # Graph-JEPA style predicted links.  These edges are inferred from
    # embedding similarity, entity overlap, and temporal adjacency.
    graph_link_prediction_enabled: bool = False
    graph_link_prediction_top_k: int = 8
    graph_link_prediction_threshold: float = 0.55
    graph_link_prediction_max_nodes: int = 200
    graph_link_prediction_weight: float = 0.5

    # Dialogue-JEPA style latent query expansion.  The predictor is optional;
    # if no trained checkpoint exists, the query vector is used unchanged.
    jepa_query_expansion_enabled: bool = False
    jepa_model_path: str = "./data/dialogue_jepa.pt"
    jepa_blend_weight: float = 0.5
    jepa_query_expansion_fallback: bool = True
    jepa_pseudo_relevance_seed_k: int = 20
    jepa_pseudo_relevance_weight: float = 0.5

    # Async Add processing.  When enabled, /add writes raw searchable records
    # synchronously and a worker later builds structured Gold, entities, and
    # graph edges.
    async_processing_enabled: bool = False
    async_worker_enabled: bool = False
    async_worker_poll_interval: float = 1.0

    # Retrieval controls.
    retrieval_candidate_k: int = 200
    retrieval_dense_k: int = 100
    retrieval_sparse_k: int = 100
    max_return_items: int = 100
    max_content_chars: int = 4000

    @property
    def database_file(self) -> Path:
        return Path(self.database_path).expanduser().resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
