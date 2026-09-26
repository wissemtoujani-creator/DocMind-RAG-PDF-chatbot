"""PDF ingestion.

Loading is intentionally defensive: a production service must reject malformed,
password-protected, or oversized uploads before they reach the chunker, and must
distinguish "this PDF has no extractable text" from "this PDF is broken" — the
two need very different messages for the user.
"""

from __future__ import annotations

from pathlib import Path

from app.core.errors import PayloadTooLargeError, UnsupportedFileTypeError, ValidationError
from app.core.logging import get_logger
from app.services.rag.schemas import Page

logger = get_logger(__name__)

PDF_MAGIC = b"%PDF-"


def validate_upload(filename: str | None, size_bytes: int, max_bytes: int) -> str:
    """Check extension and size before the file is read in full."""
    name = (filename or "").strip()
    if not name.lower().endswith(".pdf"):
        raise UnsupportedFileTypeError(f"{name or 'file'} is not a PDF.")
    if size_bytes <= 0:
        raise ValidationError("The uploaded file is empty.")
    if size_bytes > max_bytes:
        limit_mb = round(max_bytes / 1_048_576, 1)
        raise PayloadTooLargeError(f"File exceeds the {limit_mb} MB limit.")
    return Path(name).name


def looks_like_pdf(head: bytes) -> bool:
    return head.lstrip()[: len(PDF_MAGIC)] == PDF_MAGIC


def extract_pages(pdf_path: Path) -> list[Page]:
    """Extract text page by page, tolerating pages that raise individually."""
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(str(pdf_path))
    except PdfReadError as exc:
        raise ValidationError(f"Could not read the PDF: {exc}") from exc
    except Exception as exc:
        raise ValidationError(f"Could not read the PDF: {exc}") from exc

    if reader.is_encrypted:
        # Many "encrypted" PDFs merely have an empty owner password; try it
        # before giving up so those still work.
        try:
            if reader.decrypt("") == 0:
                raise ValidationError("This PDF is password protected.")
        except ValidationError:
            raise
        except Exception as exc:
            raise ValidationError(f"This PDF is password protected: {exc}") from exc

    pages: list[Page] = []
    failed = 0
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            logger.warning("page.extraction_failed", page=index, document=pdf_path.name)
            failed += 1
            text = ""
        pages.append(Page(number=index, text=text))

    if not pages:
        raise ValidationError("The PDF contains no pages.")
    if all(page.is_empty for page in pages):
        raise ValidationError(
            "No text could be extracted. The PDF is likely a scanned image; "
            "run OCR before uploading."
        )
    if failed:
        logger.warning("pages.partial_failure", failed=failed, total=len(pages))

    logger.info(
        "pdf.extracted",
        document=pdf_path.name,
        pages=len(pages),
        chars=sum(len(page.text) for page in pages),
    )
    return pages
