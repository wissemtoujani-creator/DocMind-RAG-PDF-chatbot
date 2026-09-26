"""Application settings, loaded once from the environment.

Every tunable the RAG pipeline has lives here so that behaviour is driven by
configuration rather than by hard-coded literals in the service layer.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # ------------------------------------------------------------------ app
    app_name: str = "DocMind"
    app_version: str = "0.1.0"
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="console", description="'console' or 'json'")
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # ------------------------------------------------------------- providers
    groq_api_key: str | None = Field(default=None)
    llm_model: str = Field(
        default="qwen/qwen3.8-27b",
        description="Groq chat model id. 'openai/gpt-oss-120b' is a reasoning model "
        "and costs latency in tokens that never reach the answer.",
    )
    llm_temperature: float = Field(default=0.2)
    llm_timeout_s: float = Field(default=60.0)
    llm_max_retries: int = Field(default=2)

    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    embedding_device: str = Field(default="cpu")
    embedding_batch_size: int = Field(default=32)

    # ------------------------------------------------------------- retrieval
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)
    top_k: int = Field(default=5)
    history_turns: int = Field(default=5)
    max_question_chars: int = Field(default=2000)
    max_upload_bytes: int = Field(default=25 * 1024 * 1024)
    preview_chars: int = Field(default=200)

    # ----------------------------------------------------------- vector store
    vector_store_backend: str = Field(default="chroma")
    chroma_persist_dir: Path = Field(default=PROJECT_ROOT / "chroma_db")
    chroma_collection: str = Field(default="pdf_documents")

    # ------------------------------------------------------------ filesystem
    storage_dir: Path = Field(default=PROJECT_ROOT / "storage")
    static_dir: Path = Field(default=PROJECT_ROOT / "app" / "static")

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        return value.upper()

    @field_validator("log_format")
    @classmethod
    def _check_log_format(cls, value: str) -> str:
        normalised = value.lower()
        if normalised not in {"console", "json"}:
            raise ValueError("log_format must be 'console' or 'json'")
        return normalised

    @field_validator("vector_store_backend")
    @classmethod
    def _check_backend(cls, value: str) -> str:
        normalised = value.lower()
        if normalised not in {"chroma", "pgvector"}:
            raise ValueError("vector_store_backend must be 'chroma' or 'pgvector'")
        return normalised

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in {"production", "prod"}

    @property
    def effective_cors_origins(self) -> list[str]:
        """Wildcards are unsafe in production; fall back to a closed default."""
        if self.is_production and "*" in self.cors_origins:
            return []
        return self.cors_origins


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
