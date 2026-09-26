"""Document upload and lifecycle endpoints."""

from __future__ import annotations

import asyncio
import shutil
import time

from fastapi import APIRouter, File, UploadFile, status

from app.api.deps import DocumentRepoDep, PipelineDep, SettingsDep
from app.api.v1.schemas import (
    DocumentListResponse,
    DocumentResponse,
    ErrorResponse,
    UploadResponse,
)
from app.core.errors import UnsupportedFileTypeError, ValidationError
from app.core.logging import get_logger
from app.repositories.documents import content_hash
from app.services.rag.loader import looks_like_pdf, validate_upload
from app.services.rag.schemas import DocumentRecord, DocumentStatus, new_id

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

_ERROR_RESPONSES = {
    404: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
    415: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
}


def _to_response(record: DocumentRecord) -> DocumentResponse:
    return DocumentResponse(**record.model_dump())


@router.post(
    "",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a PDF",
    responses=_ERROR_RESPONSES,
)
async def upload_document(
    settings: SettingsDep,
    pipeline: PipelineDep,
    documents: DocumentRepoDep,
    file: UploadFile = File(...),
) -> UploadResponse:
    filename = validate_upload(file.filename, file.size or 0, settings.max_upload_bytes)

    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    document_id = new_id("doc")
    staging_dir = settings.storage_dir / document_id
    staging_dir.mkdir(parents=True, exist_ok=True)
    target = staging_dir / filename

    started = time.perf_counter()
    try:
        with target.open("wb") as buffer:
            await asyncio.to_thread(shutil.copyfileobj, file.file, buffer)
        payload = target.read_bytes()

        if not looks_like_pdf(payload[:1024]):
            raise UnsupportedFileTypeError(
                "The file does not look like a PDF despite its extension."
            )
        filename = validate_upload(filename, len(payload), settings.max_upload_bytes)

        record = DocumentRecord(
            id=document_id,
            filename=filename,
            status=DocumentStatus.INDEXING,
            size_bytes=len(payload),
            content_hash=content_hash(payload),
        )
        await documents.create(record)

        # Embedding and LLM work is blocking and IO-bound: never on the event loop.
        result = await asyncio.to_thread(
            pipeline.ingest,
            pdf_path=target,
            document_id=document_id,
            filename=filename,
            size_bytes=len(payload),
        )
        await documents.update(document_id, **result.document.model_dump())
    except Exception as exc:
        await documents.delete(document_id)
        logger.warning("upload.failed", filename=filename, error=str(exc))
        raise
    finally:
        await file.close()
        shutil.rmtree(staging_dir, ignore_errors=True)

    return UploadResponse(
        document=_to_response(result.document),
        document_id=document_id,
        filename=filename,
        pages=result.document.pages,
        chunks=result.document.chunks,
        elapsed_s=round(time.perf_counter() - started, 2),
    )


@router.get("", response_model=DocumentListResponse, summary="List indexed documents")
async def list_documents(documents: DocumentRepoDep) -> DocumentListResponse:
    records = await documents.list()
    return DocumentListResponse(
        documents=[_to_response(record) for record in records],
        count=len(records),
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Fetch one document",
    responses={404: {"model": ErrorResponse}},
)
async def get_document(document_id: str, documents: DocumentRepoDep) -> DocumentResponse:
    if not document_id or len(document_id) > 64:
        raise ValidationError("Malformed document id.")
    return _to_response(await documents.require(document_id))


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and its embeddings",
    responses={404: {"model": ErrorResponse}},
)
async def delete_document(
    document_id: str,
    pipeline: PipelineDep,
    documents: DocumentRepoDep,
) -> None:
    await documents.require(document_id)
    removed = await asyncio.to_thread(pipeline.delete, document_id)
    await documents.delete(document_id)
    logger.info("document.deleted", document_id=document_id, chunks=removed)
