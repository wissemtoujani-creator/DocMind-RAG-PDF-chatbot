"""Aggregates the v1 routers.

Versioning is path-based (``/api/v1``) so a v2 can ship alongside v1 while old
clients keep resolving.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.routes import chat, documents, health, stats

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(documents.router)
api_router.include_router(chat.router)
api_router.include_router(stats.router)
