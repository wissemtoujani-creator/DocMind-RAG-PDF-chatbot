"""ASGI middleware: request correlation, access logging, security headers.

These are implemented as raw ASGI middleware rather than Starlette's
``BaseHTTPMiddleware``. The latter runs the downstream app in a separate task,
which breaks ``contextvars`` propagation — precisely the mechanism the logging
layer relies on to attach a ``request_id`` to every line.
"""

from __future__ import annotations

import time
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core import context
from app.core.logging import get_logger

logger = get_logger(__name__)

REQUEST_ID_HEADER = b"x-request-id"

_SECURITY_HEADERS: tuple[tuple[bytes, bytes], ...] = (
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (b"cross-origin-opener-policy", b"same-origin"),
)


def _append_header(headers: list[tuple[bytes, bytes]], key: bytes, value: bytes) -> None:
    """Set a header on a raw ASGI header list, which is ``bytes`` throughout.

    Starlette's ``MutableHeaders`` is string-based, so it cannot be used on an
    outgoing ``http.response.start`` message.
    """
    lowered = key.lower()
    for existing, _ in headers:
        if existing.lower() == lowered:
            return
    headers.append((key, value))


class RequestContextMiddleware:
    """Assign a request id, time the request, and log the outcome.

    An inbound ``X-Request-ID`` is honoured so a trace can span the API gateway
    and this service; anything unparseable is replaced.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        inbound = headers.get(REQUEST_ID_HEADER, b"").decode("latin-1").strip()
        request_id = _sanitise(inbound) or context.new_request_id()

        token = context.set_request_id(request_id)
        request = Request(scope, receive=receive)
        started = time.perf_counter()
        status_holder: dict[str, Any] = {"status": 500, "error": None}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                _append_header(
                    message.setdefault("headers", []), REQUEST_ID_HEADER, request_id.encode()
                )
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            status_holder["error"] = exc
            raise
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            context.reset_request_id(token)
            self._log(request, status_holder, duration_ms)

    @staticmethod
    def _log(request: Request, status_holder: dict[str, Any], duration_ms: float) -> None:
        status = status_holder["status"]
        error = status_holder["error"]
        event = {
            "http.method": request.method,
            "url.path": request.url.path,
            "http.status_code": status,
            "duration_ms": duration_ms,
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
        }
        if error is not None:
            logger.error("request_failed", exc_info=error, **event)
        elif status >= 500:
            logger.error("request_completed", **event)
        elif status >= 400:
            logger.warning("request_completed", **event)
        else:
            logger.info("request_completed", **event)


class SecurityHeadersMiddleware:
    """Attach conservative security headers to every response."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                for key, value in _SECURITY_HEADERS:
                    _append_header(headers, key, value)
            await send(message)

        await self.app(scope, receive, send_wrapper)


class BodySizeLimitMiddleware:
    """Reject oversized request bodies before they are buffered to disk.

    Applied to JSON endpoints only; multipart uploads are bounded separately at
    the route level where the file is already being streamed to a temp file.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        content_type = dict(scope.get("headers", [])).get(b"content-type", b"")
        is_json = content_type.startswith(b"application/json")
        if not is_json:
            await self.app(scope, receive, send)
            return

        total = 0

        async def counting_receive() -> Message:
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self.max_bytes:
                    from app.core.errors import PayloadTooLargeError

                    response = JSONResponse(
                        status_code=413,
                        content={
                            "error": {
                                "code": PayloadTooLargeError.code,
                                "message": "Request body is too large.",
                            }
                        },
                    )
                    await response(scope, receive, send)
                    raise _ClientDisconnect
            return message

        try:
            await self.app(scope, counting_receive, send)
        except _ClientDisconnect:
            return


class _ClientDisconnect(Exception):
    """Internal signal that the response has already been written."""


def _sanitise(value: str) -> str | None:
    """Accept only short, printable, opaque ids to avoid log/header injection."""
    if not value or len(value) > 64:
        return None
    if not all(char.isalnum() or char in "-_" for char in value):
        return None
    return value
