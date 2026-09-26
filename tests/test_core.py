"""Configuration and error-taxonomy behaviour."""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from app.core.config import Settings
from app.core.errors import AppError, ProviderError, ProviderRateLimited
from app.services.rag.generator import _translate


class TestSettings:
    def test_reads_values_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TOP_K", "9")
        monkeypatch.setenv("LLM_MODEL", "some/other-model")

        settings = Settings(_env_file=None)

        assert settings.top_k == 9
        assert settings.llm_model == "some/other-model"

    def test_parses_list_from_json_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CORS_ORIGINS", '["https://app.example.com"]')

        assert Settings(_env_file=None).cors_origins == ["https://app.example.com"]

    def test_log_level_is_normalised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "debug")
        assert Settings(_env_file=None).log_level == "DEBUG"

    @pytest.mark.parametrize("value", ["yaml", "text", ""])
    def test_rejects_unknown_log_format(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv("LOG_FORMAT", value)
        with pytest.raises(PydanticValidationError):
            Settings(_env_file=None)

    @pytest.mark.parametrize("value", ["milvus", "faiss", ""])
    def test_rejects_unknown_vector_backend(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("VECTOR_STORE_BACKEND", value)
        with pytest.raises(PydanticValidationError):
            Settings(_env_file=None)

    def test_backend_choice_is_normalised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VECTOR_STORE_BACKEND", "PGVector")
        assert Settings(_env_file=None).vector_store_backend == "pgvector"

    def test_wildcard_cors_is_dropped_in_production(self) -> None:
        settings = Settings(environment="production", cors_origins=["*"], _env_file=None)
        assert settings.effective_cors_origins == []

    def test_explicit_cors_is_kept_in_production(self) -> None:
        settings = Settings(
            environment="production",
            cors_origins=["https://app.example.com"],
            _env_file=None,
        )
        assert settings.effective_cors_origins == ["https://app.example.com"]

    def test_wildcard_cors_is_kept_in_development(self) -> None:
        settings = Settings(environment="development", cors_origins=["*"], _env_file=None)
        assert settings.effective_cors_origins == ["*"]

    def test_get_settings_is_cached(self) -> None:
        from app.core.config import get_settings

        get_settings.cache_clear()
        assert get_settings() is get_settings()


class TestErrorTaxonomy:
    def test_base_error_defaults(self) -> None:
        error = AppError()
        assert error.status_code == 500
        assert error.code == "internal_error"
        assert error.details == {}

    def test_error_accepts_overrides(self) -> None:
        error = AppError("custom message", code="custom_code", details={"field": "x"})
        assert error.message == "custom message"
        assert error.code == "custom_code"
        assert error.details == {"field": "x"}

    def test_rate_limit_is_a_provider_error(self) -> None:
        assert issubclass(ProviderRateLimited, ProviderError)
        assert ProviderRateLimited().status_code == 429
        assert ProviderRateLimited().code == "provider_rate_limited"


class TestProviderErrorTranslation:
    @pytest.mark.parametrize(
        "message",
        [
            "Error code: 429 - rate limit reached",
            "Too many requests",
            "rate_limit_exceeded",
            "RateLimitError",
        ],
    )
    def test_maps_rate_limit_messages(self, message: str) -> None:
        assert isinstance(_translate(Exception(message)), ProviderRateLimited)

    def test_maps_rate_limit_by_class_name(self) -> None:
        class RateLimitError(Exception):
            pass

        assert isinstance(_translate(RateLimitError("slow down")), ProviderRateLimited)

    def test_maps_everything_else_to_provider_error(self) -> None:
        error = _translate(Exception("Error code: 500 - internal server error"))
        assert isinstance(error, ProviderError)
        assert not isinstance(error, ProviderRateLimited)

    def test_preserves_original_message(self) -> None:
        original = "Error code: 401 - invalid api key"
        assert original in str(_translate(Exception(original)))
