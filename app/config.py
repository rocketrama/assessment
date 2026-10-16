"""Service configuration, read from environment variables (and .env locally)."""

from datetime import date
from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Data (TASK.pdf §10.2)
    corpus_dir: str = "./data/corpus"
    manifest_path: str = "./data/manifest.json"
    db_path: str = "./data/veltrix.db"
    app_today: date | None = None
    cache_dir: str = "./cache"

    # LLM
    llm_provider: str = "openai"
    llm_model: str = ""
    llm_base_url: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    nvidia_api_key: str = ""
    llm_max_concurrency: int = 4
    llm_timeout_seconds: float = 12.0
    # Sent as reasoning_effort to OpenAI-compatible reasoning models (e.g. gpt-oss); empty = not sent
    llm_reasoning_effort: str = ""
    # Time budget for the LLM calls of one /chat request; the spec allows 30 s (§9.1)
    request_budget_seconds: float = 25.0
    # Ingestion must leave /health time to reach 200 within 10 minutes (must-pass #1)
    ingestion_budget_seconds: float = 480.0

    # OCR
    vision_model: str = ""
    ocr_min_confidence: float = 70.0
    # Full path to tesseract when it is not on PATH (e.g. Windows); empty = PATH
    tesseract_cmd: str = ""

    # Retrieval / rule extraction
    embedding_provider: str = "local"
    rule_cluster_threshold: float | None = None

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    @field_validator("app_today", "rule_cluster_threshold", mode="before")
    @classmethod
    def _empty_is_none(cls, value: object) -> object:
        return None if value == "" else value

    def today(self) -> date:
        """`APP_TODAY` if set, otherwise the system date (§10.2)."""
        return self.app_today or date.today()


@lru_cache
def get_settings() -> Settings:
    return Settings()
