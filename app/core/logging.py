"""Structured logging.

Application logs and third-party library logs (uvicorn, chromadb) are rendered
through one structlog pipeline, so a single log line always has the same shape
regardless of its origin. Development gets a colourised key/value stream;
production gets one JSON object per line.

The wiring follows structlog's recommended stdlib setup: structlog logs are
forwarded into a stdlib handler carrying a
:class:`~structlog.stdlib.ProcessorFormatter`, which also normalises foreign
records (those from libraries that do not use structlog) onto the same
processors.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from app.core.config import Settings
from app.core.context import get_request_id

_configured = False

# Loggers that should be routed through the structlog renderer rather than
# keeping their own bespoke handlers.
_REDIRECTED_LOGGERS = (
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
    "fastapi",
    "chromadb",
    "sentence_transformers",
    "httpx",
)


def _inject_request_id(_logger: Any, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    request_id = get_request_id()
    if request_id and "request_id" not in event_dict:
        event_dict["request_id"] = request_id
    return event_dict


def _shared_processors() -> list[Any]:
    """Processors applied to every record, ours or a library's."""
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        _inject_request_id,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]


def configure_logging(settings: Settings) -> None:
    """Install the structlog + stdlib processor chain. Idempotent."""
    global _configured
    if _configured:
        return

    level = getattr(logging, settings.log_level, logging.INFO)
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if settings.log_format == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    shared = _shared_processors()
    structlog.configure(
        processors=[
            *shared,
            # Hand off to the stdlib handler below, which owns the rendering.
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                renderer,
            ],
        )
    )

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    for name in _REDIRECTED_LOGGERS:
        stdlib_logger = logging.getLogger(name)
        stdlib_logger.handlers.clear()
        stdlib_logger.propagate = True
        stdlib_logger.setLevel(level)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
