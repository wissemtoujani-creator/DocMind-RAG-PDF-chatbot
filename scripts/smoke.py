"""Manual end-to-end smoke test against the real provider stack.

Not part of the pytest suite: it needs GROQ_API_KEY and downloads the
embedding model. Run with:  python scripts/smoke.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TMP = Path(tempfile.mkdtemp(prefix="docmind-smoke-"))
os.environ["STORAGE_DIR"] = str(TMP / "storage")
os.environ["CHROMA_PERSIST_DIR"] = str(TMP / "chroma")
os.environ["LOG_LEVEL"] = "WARNING"

from fastapi.testclient import TestClient  # noqa: E402
from tests.conftest import make_pdf  # noqa: E402

from app.main import create_app  # noqa: E402

PAGES = [
    "Refunds are available within thirty days of the invoice date.",
    "Passwords must be rotated every ninety days and never reused across systems.",
    "All customer data is encrypted at rest using AES-256 and in transit with TLS 1.3.",
    "Support is available Monday to Friday, nine to five CET.",
]


def main() -> int:
    pdf = make_pdf(TMP / "handbook.pdf", PAGES)
    with TestClient(create_app()) as client:
        print("ready   ", client.get("/api/v1/health/ready").json())

        with pdf.open("rb") as handle:
            response = client.post(
                "/api/v1/documents",
                files={"file": ("handbook.pdf", handle, "application/pdf")},
            )
        upload = response.json()
        document_id = upload["document_id"]
        print(
            "upload  ",
            response.status_code,
            f"pages={upload['pages']} chunks={upload['chunks']} in {upload['elapsed_s']}s",
        )

        for question in (
            "How long do I have to request a refund?",
            "What encryption is used?",
            "Do you support Kubernetes?",
        ):
            body = client.post(
                "/api/v1/chat",
                json={"question": question, "document_id": document_id},
            ).json()
            markers = [
                (c["marker"], f"p{c['page']}", round(c["score"] or 0, 3))
                for c in body.get("citations", [])
            ]
            print()
            print("Q:", question)
            print("A:", body.get("answer", body)[:320])
            print("   citations:", markers, "degraded:", body.get("degraded"))

        print()
        print("stats   ", client.get("/api/v1/stats").json())
        print("delete  ", client.delete(f"/api/v1/documents/{document_id}").status_code)
        print("list    ", client.get("/api/v1/documents").json()["count"], "documents remain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
