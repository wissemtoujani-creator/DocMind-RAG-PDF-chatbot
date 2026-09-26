"""Application factory and lifespan.

Collaborators are built inside the lifespan rather than at import time so that a
test can construct the app with fakes, and so a failure to load the embedding
model surfaces as a startup error instead of an import error.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1.router import api_router
from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.core.logging import configure_logging, get_logger
from app.core.middleware import (
    BodySizeLimitMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
)
from app.repositories.conversations import InMemoryConversationRepository
from app.repositories.documents import InMemoryDocumentRepository
from app.services.rag.chunking import RecursiveTextChunker
from app.services.rag.embedding import build_embedder
from app.services.rag.generator import build_generator
from app.services.rag.pipeline import RAGPipeline
from app.services.vectordb.base import build_vector_store

logger = get_logger(__name__)

DESCRIPTION = """
A retrieval-augmented generation service over PDF corpora.

* Upload a PDF — it is extracted, chunked, embedded, and indexed
* Ask questions that are answered strictly from retrieved passages
* Every claim carries a citation pointing at a page and passage
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    settings.storage_dir.mkdir(parents=True, exist_ok=True)

    embedder = build_embedder(settings)
    vector_store = build_vector_store(settings, embedder)
    app.state.pipeline = RAGPipeline(
        vector_store=vector_store,
        embedder=embedder,
        generator=build_generator(settings),
        chunker=RecursiveTextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ),
        top_k=settings.top_k,
        preview_chars=settings.preview_chars,
        history_turns=settings.history_turns,
    )
    app.state.documents = InMemoryDocumentRepository()
    app.state.conversations = InMemoryConversationRepository()

    logger.info(
        "startup.complete",
        environment=settings.environment,
        backend=settings.vector_store_backend,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
    )
    try:
        yield
    finally:
        logger.info("shutdown.complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    app.state.settings = settings

    # Middleware executes bottom-up: the request context is installed first so
    # that everything downstream, including CORS, logs with a request_id.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_question_chars * 64)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.effective_cors_origins,
        allow_credentials=not settings.is_production,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        max_age=600,
    )
    app.add_middleware(RequestContextMiddleware)

    _register_error_handlers(app)
    app.include_router(api_router, prefix=settings.api_prefix)

    _mount_frontend(app, settings)
    return app


def _register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(_request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {"code": exc.code, "message": exc.message, "details": exc.details}
            },
            headers={"Retry-After": "30"} if exc.status_code == 429 else None,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "The request payload failed validation.",
                    "details": {"errors": _jsonable_errors(exc)},
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(
        _request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": "http_error",
                    "message": str(exc.detail),
                    "details": {},
                }
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        # Never leak internals: the traceback is logged with the request id, and
        # the client gets the id to quote in a bug report.
        logger.error(
            "unhandled_exception",
            exc_info=exc,
            path=request.url.path,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "An unexpected error occurred.",
                    "details": {},
                }
            },
        )


def _jsonable_errors(exc: RequestValidationError) -> list[dict[str, object]]:
    return [
        {
            "location": ".".join(str(part) for part in error.get("loc", ())),
            "message": error.get("msg", "invalid value"),
            "type": error.get("type", "value_error"),
        }
        for error in exc.errors()
    ]


def _mount_frontend(app: FastAPI, settings: Settings) -> None:
    static_dir: Path = settings.static_dir
    if not static_dir.is_dir():
        logger.warning("frontend.missing", path=str(static_dir))
        return

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    index = static_dir / "index.html"

    @app.get("/", include_in_schema=False)
    async def serve_frontend() -> FileResponse:
        return FileResponse(index, media_type="text/html")


app = create_app()
