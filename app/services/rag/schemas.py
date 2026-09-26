"""Domain models shared between the service layer and the HTTP API.

These are deliberately transport-agnostic: the pipeline knows nothing about
FastAPI, and the router maps these models onto its own response schemas.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

UTC = timezone.utc


class DocumentStatus(str, Enum):
    PENDING = "pending"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class Page(BaseModel):
    """A single extracted page, before chunking."""

    number: int
    text: str

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class Chunk(BaseModel):
    """A retrievable unit of text with provenance back to the source page."""

    model_config = ConfigDict(frozen=True)

    id: str
    document_id: str
    text: str
    page: int
    chunk_index: int
    char_start: int | None = None
    char_end: int | None = None

    @classmethod
    def from_document(
        cls,
        *,
        document_id: str,
        page: int,
        chunk_index: int,
        text: str,
        char_start: int | None = None,
        char_end: int | None = None,
    ) -> Chunk:
        return cls(
            id=new_id("chk"),
            document_id=document_id,
            text=text,
            page=page,
            chunk_index=chunk_index,
            char_start=char_start,
            char_end=char_end,
        )


class RetrievedChunk(BaseModel):
    """A chunk returned by a similarity search, with its relevance score."""

    chunk: Chunk
    score: float | None = None


class DocumentRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    filename: str
    status: DocumentStatus = DocumentStatus.PENDING
    pages: int = 0
    chunks: int = 0
    size_bytes: int = 0
    content_hash: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    indexed_at: datetime | None = None

    @property
    def is_ready(self) -> bool:
        return self.status is DocumentStatus.READY


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant|system)$")
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Citation(BaseModel):
    """A single attribution for part of an answer."""

    marker: int
    chunk_id: str
    page: int
    source: str
    score: float | None = None
    preview: str


class RagAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    chunks_used: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    degraded: bool = Field(
        default=False,
        description="True when the answer was produced without retrieved context.",
    )


class IngestResult(BaseModel):
    document: DocumentRecord
    ingest_ms: int


class IndexStats(BaseModel):
    documents: int
    ready_documents: int
    total_pages: int
    total_chunks: int
    backend: str
