"""Request-scoped context propagated through logs via :mod:`contextvars`.

Because the ASGI stack is concurrent, plain instance attributes on loggers are
not safe. A :class:`~contextvars.ContextVar` gives every in-flight request its
own binding, which ``structlog`` reads through the processor chain in
:mod:`app.core.logging`.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar, Token

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def new_request_id() -> str:
    return uuid.uuid4().hex


def set_request_id(value: str) -> Token[str | None]:
    return _request_id.set(value)


def reset_request_id(token: Token[str | None]) -> None:
    _request_id.reset(token)


def get_request_id() -> str | None:
    return _request_id.get()
