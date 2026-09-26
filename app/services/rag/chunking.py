"""Chunking strategies.

The current implementation wraps LangChain's ``RecursiveCharacterTextSplitter``
but normalises its output into :class:`~app.services.rag.schemas.Chunk` objects
with stable ids and page provenance, so a backend swap cannot silently lose
metadata. Section-aware and semantic chunking land on top of this interface.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logging import get_logger
from app.services.rag.schemas import Chunk, Page

logger = get_logger(__name__)

DEFAULT_SEPARATORS: Sequence[str] = ("\n\n", "\n", ". ", "? ", "! ", "; ", " ", "")


class Chunker(Protocol):
    def split(self, pages: Sequence[Page], *, document_id: str) -> list[Chunk]: ...


class RecursiveTextChunker:
    """Split page text on the most semantic separator that fits.

    Splitting per page rather than across the whole document keeps provenance
    exact and stops a chunk from straddling a page break.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: Sequence[str] = DEFAULT_SEPARATORS,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=list(separators),
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, pages: Sequence[Page], *, document_id: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        for page in pages:
            text = page.text.strip()
            if not text:
                continue
            for index, piece in enumerate(self._splitter.split_text(text)):
                offset = text.find(piece)
                chunks.append(
                    Chunk.from_document(
                        document_id=document_id,
                        page=page.number,
                        chunk_index=index,
                        text=piece,
                        char_start=offset if offset >= 0 else None,
                        char_end=offset + len(piece) if offset >= 0 else None,
                    )
                )

        logger.info(
            "chunking.completed",
            document_id=document_id,
            pages=len(pages),
            chunks=len(chunks),
            avg_chars=round(sum(len(c.text) for c in chunks) / len(chunks), 1)
            if chunks
            else 0,
        )
        return chunks
