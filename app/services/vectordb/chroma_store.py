"""Chroma-backed vector store (local development default).

Documents share one collection and are separated by a ``document_id`` metadata
filter, mirroring the row-level scoping pgvector will provide via a column and a
partial index.
"""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.services.rag.schemas import Chunk, RetrievedChunk

logger = get_logger(__name__)


class ChromaVectorStore:
    backend = "chroma"

    def __init__(self, *, settings: Any, embeddings: Any) -> None:
        from langchain_chroma import Chroma

        self._settings = settings
        self._embeddings = embeddings
        self._collection_name = settings.chroma_collection
        persist_dir = str(settings.chroma_persist_dir)
        settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self._store = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_dir,
            collection_name=self._collection_name,
        )
        logger.info("vectordb.ready", backend=self.backend, persist_dir=persist_dir)

    # ------------------------------------------------------------------ #
    #  VectorStore protocol                                                #
    # ------------------------------------------------------------------ #

    def add_chunks(self, chunks: list[Chunk], *, document_id: str) -> int:
        if not chunks:
            return 0
        from langchain_core.documents import Document

        documents = [
            Document(
                id=chunk.id,
                page_content=chunk.text,
                metadata={
                    "document_id": document_id,
                    "page": chunk.page,
                    "chunk_index": chunk.chunk_index,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                },
            )
            for chunk in chunks
        ]
        ids = [document.id for document in documents]
        self._store.add_documents(documents=documents, ids=ids)
        logger.info("vectordb.chunks_added", document_id=document_id, count=len(ids))
        return len(ids)

    def similarity_search(
        self,
        query: str,
        *,
        document_id: str,
        k: int,
    ) -> list[RetrievedChunk]:
        # Raw distances are used rather than LangChain's relevance scores: the
        # latter assume a bounded similarity and emit a warning for the negative
        # cosine distances that real queries routinely produce.
        results = self._store.similarity_search_with_score(
            query,
            k=k,
            filter={"document_id": document_id},
        )
        retrieved: list[RetrievedChunk] = []
        for document, distance in results:
            page = document.metadata.get("page", 0)
            chunk_index = int(document.metadata.get("chunk_index", 0))
            retrieved.append(
                RetrievedChunk(
                    chunk=Chunk(
                        id=document.id or "",
                        document_id=document_id,
                        text=document.page_content,
                        page=int(page) if page is not None else 0,
                        chunk_index=chunk_index,
                        char_start=document.metadata.get("char_start"),
                        char_end=document.metadata.get("char_end"),
                    ),
                    # Chroma's default space is cosine, so distance is 1 - cosine
                    # similarity. Keep the raw cosine so scores stay comparable
                    # with the pgvector backend, and do not clamp.
                    score=round(1.0 - float(distance), 6) if distance is not None else None,
                )
            )
        return retrieved

    def delete_document(self, document_id: str) -> int:
        existing = self._store.get(where={"document_id": document_id})
        ids = existing.get("ids") or []
        if ids:
            self._store.delete(ids=ids)
        logger.info("vectordb.document_deleted", document_id=document_id, chunks=len(ids))
        return len(ids)

    def count(self, document_id: str | None = None) -> int:
        if document_id is None:
            return self._store._collection.count()
        return self._store._collection.count(where={"document_id": document_id})

    def health(self) -> bool:
        try:
            self._store._collection.count()
        except Exception as exc:
            logger.warning("vectordb.health_failed", exc_info=exc)
            return False
        return True
