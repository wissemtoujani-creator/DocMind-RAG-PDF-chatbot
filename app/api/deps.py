"""FastAPI dependency providers.

Collaborators are constructed once in the application lifespan and resolved per
request from ``app.state``. This keeps expensive singletons (the embedding model,
the vector store connection pool) out of the module import path, which is what
lets tests swap in fakes.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.core.config import Settings
from app.repositories.conversations import ConversationRepository
from app.repositories.documents import DocumentRepository
from app.services.rag.pipeline import RAGPipeline


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_pipeline(request: Request) -> RAGPipeline:
    return request.app.state.pipeline


def get_document_repository(request: Request) -> DocumentRepository:
    return request.app.state.documents


def get_conversation_repository(request: Request) -> ConversationRepository:
    return request.app.state.conversations


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
PipelineDep = Annotated[RAGPipeline, Depends(get_pipeline)]
DocumentRepoDep = Annotated[DocumentRepository, Depends(get_document_repository)]
ConversationRepoDep = Annotated[ConversationRepository, Depends(get_conversation_repository)]
