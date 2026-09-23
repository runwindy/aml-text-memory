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
        env_file=".env",
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
