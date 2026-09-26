"""Liveness and readiness probes.

``/health`` answers as long as the process is up, so an orchestrator does not
restart a container merely because a downstream dependency blipped. ``/health/ready``
actually touches the vector store and is what should gate traffic.
"""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from app.api.deps import PipelineDep, SettingsDep
from app.api.v1.schemas import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health(settings: SettingsDep) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        environment=settings.environment,
    )


@router.get("/health/ready", response_model=HealthResponse, summary="Readiness probe")
async def ready(
    response: Response,
    settings: SettingsDep,
    pipeline: PipelineDep,
) -> HealthResponse:
    healthy = pipeline.health()
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="ok" if healthy else "degraded",
        version=settings.app_version,
        environment=settings.environment,
        vector_store=pipeline.backend,
    )
