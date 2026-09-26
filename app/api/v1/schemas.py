"""Request and response models for API v1.

Kept separate from the domain models in ``app.services.rag.schemas`` so that
renaming a field for HTTP ergonomics does not ripple into the pipeline.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.services.rag.schemas import DocumentStatus


# --------------------------------------------------------------------- errors
class ErrorBody(BaseModel):
    code: str = Field(examples=["not_found"])
    message: str = Field(examples=["Document doc_x does not exist."])
    details: dict[str, object] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorBody


# ------------------------------------------------------------------ documents
class DocumentResponse(BaseModel):
    id: str
    filename: str
    status: DocumentStatus
    pages: int
    chunks: int
    size_bytes: int
    content_hash: str | None = None
    error: str | None = None
    created_at: datetime
    indexed_at: datetime | None = None


class UploadResponse(BaseModel):
    """Superset of the legacy ``/upload`` payload so old clients keep working."""

    success: bool = True
    document: DocumentResponse
    document_id: str
    filename: str
    pages: int
    chunks: int
    elapsed_s: float


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    count: int


# ----------------------------------------------------------------------- chat
class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_id: str | None = Field(
        default=None,
        description="Target document. Defaults to the most recently indexed one.",
    )
    conversation_id: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)


class CitationResponse(BaseModel):
    marker: int
    chunk_id: str
    page: int
    source: str
    score: float | None = None
    preview: str


class ChatMessageResponse(BaseModel):
    role: str
    content: str


class ChatResponse(BaseModel):
    answer: str
    document_id: str
    conversation_id: str
    citations: list[CitationResponse]
    sources: list[CitationResponse] = Field(
        description="Alias of citations, retained for legacy clients."
    )
    chat_history: list[ChatMessageResponse]
    latency_s: float
    degraded: bool = False
    error: bool = False


# --------------------------------------------------------------------- system
class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    vector_store: str | None = None


class StatsResponse(BaseModel):
    documents: int
    ready_documents: int
    total_pages: int
    total_chunks: int
    backend: str
