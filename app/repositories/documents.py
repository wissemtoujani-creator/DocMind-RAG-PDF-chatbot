"""Document metadata persistence.

The vector store owns chunk vectors; this owns the document row (filename, page
and chunk counts, status, content hash). Keeping them apart is what allows a
document to be listed, authenticated, and deleted without touching embeddings.

The in-memory implementation below is process-local and is replaced by the
Postgres implementation in the next milestone. Both satisfy the same protocol so
nothing above this layer changes.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Protocol

from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.services.rag.schemas import DocumentRecord, DocumentStatus

logger = get_logger(__name__)


def content_hash(data: bytes) -> str:
    """Stable content fingerprint used to deduplicate re-uploads."""
    return hashlib.sha256(data).hexdigest()


class DocumentRepository(Protocol):
    async def create(self, record: DocumentRecord) -> DocumentRecord: ...

    async def get(self, document_id: str) -> DocumentRecord | None: ...

    async def require(self, document_id: str) -> DocumentRecord: ...

    async def list(self, *, owner_id: str | None = None) -> list[DocumentRecord]: ...

    async def update(self, document_id: str, **changes: object) -> DocumentRecord: ...

    async def delete(self, document_id: str) -> None: ...

    async def stats(self) -> dict[str, int]: ...


class InMemoryDocumentRepository:
    """Async-safe in-process store. Guarded by a lock for concurrent uploads."""

    def __init__(self) -> None:
        self._records: dict[str, DocumentRecord] = {}
        self._lock = asyncio.Lock()

    async def create(self, record: DocumentRecord) -> DocumentRecord:
        async with self._lock:
            self._records[record.id] = record
        logger.info("repository.document_created", document_id=record.id)
        return record

    async def get(self, document_id: str) -> DocumentRecord | None:
        return self._records.get(document_id)

    async def require(self, document_id: str) -> DocumentRecord:
        record = self._records.get(document_id)
        if record is None:
            raise NotFoundError(f"Document {document_id} does not exist.")
        return record

    async def list(self, *, owner_id: str | None = None) -> list[DocumentRecord]:
        records = list(self._records.values())
        if owner_id is not None:
            records = [r for r in records if getattr(r, "owner_id", None) in (None, owner_id)]
        return sorted(records, key=lambda r: r.created_at, reverse=True)

    async def update(self, document_id: str, **changes: object) -> DocumentRecord:
        record = await self.require(document_id)
        updated = record.model_copy(update=changes)
        async with self._lock:
            self._records[document_id] = updated
        return updated

    async def delete(self, document_id: str) -> None:
        async with self._lock:
            self._records.pop(document_id, None)
        logger.info("repository.document_deleted", document_id=document_id)

    async def stats(self) -> dict[str, int]:
        records = list(self._records.values())
        return {
            "documents": len(records),
            "ready_documents": sum(1 for r in records if r.status is DocumentStatus.READY),
            "total_pages": sum(r.pages for r in records),
            "total_chunks": sum(r.chunks for r in records),
        }
