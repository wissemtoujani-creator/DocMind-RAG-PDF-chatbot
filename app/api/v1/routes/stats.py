"""Corpus statistics."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import DocumentRepoDep, PipelineDep
from app.api.v1.schemas import StatsResponse

router = APIRouter(tags=["system"])


@router.get("/stats", response_model=StatsResponse, summary="Corpus statistics")
async def get_stats(
    documents: DocumentRepoDep,
    pipeline: PipelineDep,
) -> StatsResponse:
    counts = await documents.stats()
    return StatsResponse(**counts, backend=pipeline.backend)
