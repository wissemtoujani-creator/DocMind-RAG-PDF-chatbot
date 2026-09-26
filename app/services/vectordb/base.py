"""Vector store abstraction.

The rest of the application depends only on :class:`VectorStore`. That keeps the
Chroma implementation (local development) interchangeable with the pgvector
implementation (production) without touching the pipeline.

Every method is document-scoped: one corpus can hold many documents and deleting
one must never affect another.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.services.rag.schemas import Chunk, RetrievedChunk


@runtime_checkable
class VectorStore(Protocol):
    backend: str

    def add_chunks(self, chunks: list[Chunk], *, document_id: str) -> int: ...

    def similarity_search(
        self,
        query: str,
        *,
        document_id: str,
        k: int,
    ) -> list[RetrievedChunk]: ...

    def delete_document(self, document_id: str) -> int: ...

    def count(self, document_id: str | None = None) -> int: ...

    def health(self) -> bool: ...


def build_vector_store(settings, embeddings) -> VectorStore:
    """Select a backend from configuration."""
    from app.services.vectordb.chroma_store import ChromaVectorStore

    if settings.vector_store_backend == "pgvector":
        from app.services.vectordb.pgvector_store import PgVectorStore

        return PgVectorStore(settings=settings, embeddings=embeddings)
    return ChromaVectorStore(settings=settings, embeddings=embeddings)
