"""Answer generation.

The generator is deliberately tiny: retrieve-then-generate lives in the
pipeline, and this class only turns a rendered prompt into text. Keeping the
seam narrow is what lets the evaluation harness and the test suite run without
a live LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from app.core.errors import ProviderError, ProviderRateLimited
from app.core.logging import get_logger

logger = get_logger(__name__)

REFUSAL_MARKER = "I don't have this information in the document."


@dataclass(slots=True)
class Generation:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@runtime_checkable
class AnswerGenerator(Protocol):
    def generate(self, prompt: str) -> Generation: ...


class GroqGenerator:
    """Chat completions via Groq, with bounded retries and rate-limit mapping."""

    def __init__(
        self,
        *,
        model: str,
        temperature: float,
        api_key: str | None,
        timeout_s: float,
        max_retries: int,
    ) -> None:
        from langchain_groq import ChatGroq

        self._model = model
        # ``max_retries`` already applies exponential backoff internally, so no
        # separate retry policy is layered on top.
        self._client = ChatGroq(
            model_name=model,
            temperature=temperature,
            groq_api_key=api_key,
            request_timeout=timeout_s,
            max_retries=max_retries,
            default_headers={"x-client": "docmind"},
        )
        logger.info("generator.ready", model=model, configured=bool(api_key))

    def generate(self, prompt: str) -> Generation:
        try:
            response = self._client.invoke(prompt)
        except Exception as exc:
            raise _translate(exc) from exc

        metadata = getattr(response, "usage_metadata", None) or {}
        text = (getattr(response, "content", "") or "").strip()
        return Generation(
            text=text,
            prompt_tokens=metadata.get("input_tokens"),
            completion_tokens=metadata.get("output_tokens"),
        )


def _translate(exc: Exception) -> ProviderError:
    """Map provider exceptions onto the application's error taxonomy.

    Both the exception class name and the message are inspected: providers are
    inconsistent about which one carries the signal, and a rate limit that
    arrives as a generic 502 is exactly the case worth catching.
    """
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    squashed = message.replace(" ", "").replace("_", "")
    rate_limited = (
        "ratelimit" in name
        or "ratelimit" in squashed
        or "429" in message
        or "too many requests" in message
    )
    if rate_limited:
        return ProviderRateLimited(str(exc))
    return ProviderError(f"Generation failed: {exc}")


def build_generator(settings: Any) -> AnswerGenerator:
    if not settings.groq_api_key:
        logger.warning("generator.unconfigured", reason="GROQ_API_KEY is not set")
    return GroqGenerator(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.groq_api_key,
        timeout_s=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
    )
