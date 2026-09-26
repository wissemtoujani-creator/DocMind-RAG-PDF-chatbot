"""Retrieval-augmented generation orchestration.

The pipeline is stateless: all per-corpus state lives in the
:class:`~app.services.vectordb.base.VectorStore`, keyed by ``document_id``. That
is what makes concurrent users possible — the previous implementation held a
single process-wide ``RAGPipeline`` and therefore served exactly one document to
the entire server.
"""

from __future__ import annotations

import time
from pathlib import Path

from app.core.logging import get_logger
from app.services.rag.chunking import RecursiveTextChunker
from app.services.rag.embedding import Embedder
from app.services.rag.generator import AnswerGenerator
from app.services.rag.loader import extract_pages
from app.services.rag.prompts import build_answer_prompt, build_history_block
from app.services.rag.schemas import (
    ChatMessage,
    Citation,
    DocumentRecord,
    DocumentStatus,
    IngestResult,
    RagAnswer,
    RetrievedChunk,
)
from app.services.vectordb.base import VectorStore

logger = get_logger(__name__)

NO_CONTEXT_ANSWER = "I don't have this information in the document."


class RAGPipeline:
    def __init__(
        self,
        *,
        vector_store: VectorStore,
        embedder: Embedder,
        generator: AnswerGenerator,
        chunker: RecursiveTextChunker,
        top_k: int = 5,
        preview_chars: int = 200,
        history_turns: int = 5,
    ) -> None:
        self._vectors = vector_store
        self._embedder = embedder
        self._generator = generator
        self._chunker = chunker
        self._top_k = top_k
        self._preview_chars = preview_chars
        self._history_turns = history_turns
        logger.info(
            "pipeline.ready",
            backend=vector_store.backend,
            top_k=top_k,
            chunk_size=chunker.chunk_size,
            chunk_overlap=chunker.chunk_overlap,
        )

    # ------------------------------------------------------------------ #
    #  Ingestion                                                           #
    # ------------------------------------------------------------------ #

    def ingest(
        self,
        *,
        pdf_path: Path,
        document_id: str,
        filename: str,
        size_bytes: int,
    ) -> IngestResult:
        started = time.perf_counter()
        pages = extract_pages(pdf_path)
        chunks = self._chunker.split(pages, document_id=document_id)
        if not chunks:
            raise ValueError("Chunking produced no chunks; the document is unusable.")

        self._vectors.add_chunks(chunks, document_id=document_id)
        ingest_ms = int((time.perf_counter() - started) * 1000)

        record = DocumentRecord(
            id=document_id,
            filename=filename,
            status=DocumentStatus.READY,
            pages=len(pages),
            chunks=len(chunks),
            size_bytes=size_bytes,
        )
        logger.info(
            "ingest.completed",
            document_id=document_id,
            filename=filename,
            pages=len(pages),
            chunks=len(chunks),
            ingest_ms=ingest_ms,
        )
        return IngestResult(document=record, ingest_ms=ingest_ms)

    # ------------------------------------------------------------------ #
    #  Retrieval                                                           #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        question: str,
        *,
        document_id: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        k = top_k or self._top_k
        return self._vectors.similarity_search(question, document_id=document_id, k=k)

    # ------------------------------------------------------------------ #
    #  Generation                                                          #
    # ------------------------------------------------------------------ #

    def answer(
        self,
        question: str,
        *,
        document_id: str,
        filename: str,
        history: list[ChatMessage] | None = None,
        top_k: int | None = None,
    ) -> RagAnswer:
        chunks = self.retrieve(question, document_id=document_id, top_k=top_k)

        if not chunks:
            logger.info("answer.no_context", document_id=document_id)
            return RagAnswer(answer=NO_CONTEXT_ANSWER, citations=[], chunks_used=0, degraded=True)

        history_block = build_history_block(
            [(message.role, message.content) for message in (history or [])][
                -self._history_turns * 2 :
            ]
        )
        prompt = build_answer_prompt(question, chunks)
        if history_block:
            prompt = f"{history_block}{prompt}"

        generation = self._generator.generate(prompt)
        return RagAnswer(
            answer=generation.text or NO_CONTEXT_ANSWER,
            citations=self.build_citations(chunks, filename=filename),
            chunks_used=len(chunks),
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
        )

    def build_citations(
        self,
        chunks: list[RetrievedChunk],
        *,
        filename: str,
    ) -> list[Citation]:
        """One citation per retrieved passage, numbered to match the prompt."""
        citations: list[Citation] = []
        for position, retrieved in enumerate(chunks, start=1):
            text = retrieved.chunk.text
            preview = text[: self._preview_chars]
            citations.append(
                Citation(
                    marker=position,
                    chunk_id=retrieved.chunk.id,
                    page=retrieved.chunk.page,
                    source=filename,
                    score=retrieved.score,
                    preview=preview + ("…" if len(text) > self._preview_chars else ""),
                )
            )
        return citations

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    @property
    def backend(self) -> str:
        return self._vectors.backend

    def delete(self, document_id: str) -> int:
        return self._vectors.delete_document(document_id)

    def health(self) -> bool:
        return self._vectors.health()
