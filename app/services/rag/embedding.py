"""Embedding provider.

Isolated behind a protocol so evaluation runs and unit tests can substitute a
deterministic embedder without downloading model weights or reaching the network.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from app.core.errors import ProviderError
from app.core.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Local sentence-transformers embedder, normalised for cosine similarity."""

    def __init__(self, *, model_name: str, device: str, batch_size: int) -> None:
        from langchain_huggingface import HuggingFaceEmbeddings

        self._model_name = model_name
        self._embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
        )
        logger.info("embeddings.ready", model=model_name, device=device)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            return self._embeddings.embed_documents(texts)
        except Exception as exc:
            raise ProviderError(f"Embedding failed: {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        try:
            return self._embeddings.embed_query(text)
        except Exception as exc:
            raise ProviderError(f"Embedding failed: {exc}") from exc


def build_embedder(settings: Any) -> Embedder:
    return SentenceTransformerEmbedder(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
