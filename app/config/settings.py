from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SUPPORT_AGENT_", extra="ignore")

    app_name: str = "support-agentic-rag"
    environment: str = "local"
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_timeout_s: float = 12.0
    llm_retry_count: int = 2
    retrieval_top_k: int = 4
    confidence_threshold: float = 0.62
    max_agent_steps: int = 4
    enable_reranking: bool = False
    prompt_version: str = "v1"
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
