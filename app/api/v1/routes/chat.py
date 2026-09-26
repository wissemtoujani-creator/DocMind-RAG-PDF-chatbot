"""Conversational question answering."""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, status

from app.api.deps import ConversationRepoDep, DocumentRepoDep, PipelineDep, SettingsDep
from app.api.v1.schemas import (
    ChatMessageResponse,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
)
from app.core.errors import NoDocumentError, NotFoundError, ValidationError
from app.core.logging import get_logger
from app.services.rag.schemas import ChatMessage, new_id

logger = get_logger(__name__)

router = APIRouter(tags=["chat"])

_ERROR_RESPONSES = {
    404: {"model": ErrorResponse},
    409: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    502: {"model": ErrorResponse},
}


async def _resolve_document(documents: DocumentRepoDep, document_id: str | None):
    """Fall back to the newest ready document when none is named."""
    if document_id:
        record = await documents.require(document_id)
        if not record.is_ready:
            raise NotFoundError(
                f"Document {document_id} is {record.status.value}, not ready for queries."
            )
        return record

    candidates = await documents.list()
    ready = [record for record in candidates if record.is_ready]
    if not ready:
        raise NoDocumentError()
    return ready[0]


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a grounded question about an indexed document",
    responses=_ERROR_RESPONSES,
)
async def ask(
    body: ChatRequest,
    settings: SettingsDep,
    pipeline: PipelineDep,
    documents: DocumentRepoDep,
    conversations: ConversationRepoDep,
) -> ChatResponse:
    question = body.question.strip()
    if not question:
        raise ValidationError("The question cannot be blank.")
    if len(question) > settings.max_question_chars:
        raise ValidationError(
            f"The question exceeds the {settings.max_question_chars} character limit."
        )

    record = await _resolve_document(documents, body.document_id)
    conversation_id = body.conversation_id or new_id("conv")
    history = await conversations.history(
        conversation_id, limit=settings.history_turns * 2
    )

    started = time.perf_counter()
    # Retrieval and generation are blocking and IO-bound: keep them off the loop.
    result = await asyncio.to_thread(
        pipeline.answer,
        question,
        document_id=record.id,
        filename=record.filename,
        history=history,
        top_k=body.top_k,
    )
    latency_s = round(time.perf_counter() - started, 3)

    user_message = ChatMessage(role="user", content=question)
    assistant_message = ChatMessage(role="assistant", content=result.answer)
    await conversations.append(conversation_id, [user_message, assistant_message])
    updated_history = await conversations.history(
        conversation_id, limit=settings.history_turns * 2
    )

    logger.info(
        "chat.answered",
        document_id=record.id,
        conversation_id=conversation_id,
        chunks_used=result.chunks_used,
        latency_s=latency_s,
        degraded=result.degraded,
    )

    citations = [citation.model_dump() for citation in result.citations]
    return ChatResponse(
        answer=result.answer,
        document_id=record.id,
        conversation_id=conversation_id,
        citations=citations,
        sources=citations,
        chat_history=[
            ChatMessageResponse(role=m.role, content=m.content) for m in updated_history
        ],
        latency_s=latency_s,
        degraded=result.degraded,
        error=False,
    )


@router.post(
    "/chat/reset",
    status_code=status.HTTP_200_OK,
    summary="Clear a conversation's history",
)
async def reset_conversation(
    conversations: ConversationRepoDep,
    conversation_id: str,
) -> dict[str, object]:
    await conversations.clear(conversation_id)
    logger.info("chat.reset", conversation_id=conversation_id)
    return {"success": True, "conversation_id": conversation_id}
