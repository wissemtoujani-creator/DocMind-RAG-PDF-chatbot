"""Domain and API error taxonomy.

Services raise these; the handlers registered in :mod:`app.main` translate them
into a single machine-readable error envelope so clients never have to parse a
prose ``detail`` string.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for every expected (i.e. non-bug) failure."""

    status_code: int = 500
    code: str = "internal_error"
    message: str = "An unexpected error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message or self.message
        self.code = code or self.code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(AppError):
    status_code = 422
    code = "validation_error"
    message = "The request payload failed validation."


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"
    message = "The requested resource does not exist."


class ConflictError(AppError):
    status_code = 409
    code = "conflict"
    message = "The request conflicts with the current state of the resource."


class DocumentNotReadyError(AppError):
    status_code = 409
    code = "document_not_ready"
    message = "The document is still being indexed."


class NoDocumentError(AppError):
    status_code = 409
    code = "no_document"
    message = "Upload a document before asking questions."


class UnsupportedFileTypeError(AppError):
    status_code = 415
    code = "unsupported_file_type"
    message = "Only PDF files are supported."


class PayloadTooLargeError(AppError):
    status_code = 413
    code = "payload_too_large"
    message = "The uploaded file exceeds the maximum allowed size."


class ProviderError(AppError):
    """An upstream LLM or embedding provider failed."""

    status_code = 502
    code = "provider_error"
    message = "An upstream model provider returned an error."


class ProviderRateLimited(ProviderError):
    status_code = 429
    code = "provider_rate_limited"
    message = "The model provider is rate limiting requests; retry shortly."
