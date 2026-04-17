from __future__ import annotations

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SUPPORT_AGENT_", extra="ignore")

    app_name: str = "support-agentic-rag"
    environment: str = "local"
    vector_store_backend: str = "sqlite"
    database_url: str = "postgresql://support:support@localhost:5432/support_agent"
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_timeout_s: float = 12.0
    llm_retry_count: int = 2
    retrieval_top_k: int = 4
    vector_store_path: str = "./data/vector_store.sqlite3"
    pgvector_dimension: int = 64
    confidence_threshold: float = 0.62
    max_agent_steps: int = 4
    enable_reranking: bool = False
    prompt_version: str = "v1"
    log_level: str = "INFO"
    allowed_tools: list[str] = Field(default_factory=lambda: ["search_docs", "create_ticket"])


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
