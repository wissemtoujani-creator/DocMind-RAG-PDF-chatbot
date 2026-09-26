"""HTTP contract tests.

These run through the real application factory and middleware stack, so they
cover request correlation, the error envelope, and status codes as a client sees
them.
"""

from __future__ import annotations

import pytest
from tests.conftest import make_pdf


@pytest.fixture
def pdf(tmp_path):
    return make_pdf(
        tmp_path / "policy.pdf",
        pages=[
            "Passwords must be rotated every ninety days.",
            "All data is encrypted at rest using AES-256.",
        ],
    )


def upload(client, pdf, name: str = "policy.pdf"):
    with pdf.open("rb") as handle:
        return client.post(
            "/api/v1/documents",
            files={"file": (name, handle, "application/pdf")},
        )


# ---------------------------------------------------------------------- health
class TestHealth:
    def test_health_is_public(self, client) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_ready_reports_backend(self, client) -> None:
        response = client.get("/api/v1/health/ready")
        assert response.status_code == 200
        assert response.json()["vector_store"] == "fake"


# ------------------------------------------------------------------- requests
class TestRequestContext:
    def test_generates_request_id_header(self, client) -> None:
        response = client.get("/api/v1/health")
        assert response.headers.get("x-request-id")

    def test_honours_inbound_request_id(self, client) -> None:
        response = client.get("/api/v1/health", headers={"X-Request-ID": "abc123"})
        assert response.headers["x-request-id"] == "abc123"

    def test_rejects_malformed_inbound_request_id(self, client) -> None:
        response = client.get("/api/v1/health", headers={"X-Request-ID": "bad id; drop"})
        assert response.headers["x-request-id"] != "bad id; drop"

    def test_sets_security_headers(self, client) -> None:
        headers = client.get("/api/v1/health").headers
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["x-frame-options"] == "DENY"


# ------------------------------------------------------------------ documents
class TestDocuments:
    def test_upload_returns_document_metadata(self, client, pdf) -> None:
        response = upload(client, pdf)

        assert response.status_code == 201
        body = response.json()
        assert body["success"] is True
        assert body["pages"] == 2
        assert body["chunks"] > 0
        assert body["document"]["status"] == "ready"
        assert body["document_id"].startswith("doc_")

    def test_upload_rejects_non_pdf(self, client, tmp_path) -> None:
        fake = tmp_path / "notes.txt"
        fake.write_text("hello")
        with fake.open("rb") as handle:
            response = client.post(
                "/api/v1/documents", files={"file": ("notes.txt", handle, "text/plain")}
            )

        assert response.status_code == 415
        assert response.json()["error"]["code"] == "unsupported_file_type"

    def test_upload_rejects_spoofed_extension(self, client, tmp_path) -> None:
        spoofed = tmp_path / "evil.pdf"
        spoofed.write_bytes(b"PK\x03\x04 definitely a zip")
        with spoofed.open("rb") as handle:
            response = client.post(
                "/api/v1/documents", files={"file": ("evil.pdf", handle, "application/pdf")}
            )

        assert response.status_code == 415

    def test_upload_rejects_empty_file(self, client) -> None:
        response = client.post(
            "/api/v1/documents", files={"file": ("empty.pdf", b"", "application/pdf")}
        )
        assert response.status_code == 422

    def test_failed_upload_leaves_no_document(self, client, tmp_path) -> None:
        broken = tmp_path / "broken.pdf"
        broken.write_bytes(b"%PDF-1.4 but truncated garbage")
        with broken.open("rb") as handle:
            response = client.post(
                "/api/v1/documents", files={"file": ("broken.pdf", handle, "application/pdf")}
            )

        assert response.status_code == 422
        assert client.get("/api/v1/documents").json()["count"] == 0

    def test_list_documents(self, client, pdf) -> None:
        upload(client, pdf)

        body = client.get("/api/v1/documents").json()

        assert body["count"] == 1
        assert body["documents"][0]["filename"] == "policy.pdf"

    def test_get_unknown_document_returns_error_envelope(self, client) -> None:
        response = client.get("/api/v1/documents/doc_missing")

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"

    def test_delete_document(self, client, pdf) -> None:
        document_id = upload(client, pdf).json()["document_id"]

        assert client.delete(f"/api/v1/documents/{document_id}").status_code == 204
        assert client.get("/api/v1/documents").json()["count"] == 0

    def test_delete_unknown_document(self, client) -> None:
        assert client.delete("/api/v1/documents/doc_missing").status_code == 404


# ----------------------------------------------------------------------- chat
class TestChat:
    def test_ask_without_document_returns_409(self, client) -> None:
        response = client.post("/api/v1/chat", json={"question": "What is the policy?"})

        assert response.status_code == 409
        assert response.json()["error"]["code"] == "no_document"

    def test_ask_returns_answer_and_citations(self, client, pdf) -> None:
        document_id = upload(client, pdf).json()["document_id"]

        response = client.post(
            "/api/v1/chat",
            json={"question": "How often are passwords rotated?", "document_id": document_id},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["answer"]
        assert body["document_id"] == document_id
        assert body["citations"]
        assert body["sources"] == body["citations"]
        assert body["latency_s"] >= 0
        assert body["error"] is False

    def test_ask_defaults_to_most_recent_document(self, client, pdf) -> None:
        document_id = upload(client, pdf).json()["document_id"]

        body = client.post("/api/v1/chat", json={"question": "Is data encrypted?"}).json()

        assert body["document_id"] == document_id

    def test_ask_rejects_blank_question(self, client, pdf) -> None:
        upload(client, pdf)
        response = client.post("/api/v1/chat", json={"question": "   "})
        assert response.status_code == 422

    def test_ask_rejects_oversized_question(self, client, pdf) -> None:
        upload(client, pdf)
        response = client.post("/api/v1/chat", json={"question": "x" * 5000})
        assert response.status_code == 422

    def test_ask_against_unknown_document(self, client, pdf) -> None:
        upload(client, pdf)
        response = client.post(
            "/api/v1/chat", json={"question": "hello", "document_id": "doc_missing"}
        )
        assert response.status_code == 404

    def test_conversation_history_accumulates_per_conversation(self, client, pdf) -> None:
        document_id = upload(client, pdf).json()["document_id"]
        payload = {"question": "How often?", "document_id": document_id}

        first = client.post("/api/v1/chat", json=payload).json()
        second = client.post(
            "/api/v1/chat",
            json={**payload, "conversation_id": first["conversation_id"]},
        ).json()

        assert second["conversation_id"] == first["conversation_id"]
        assert len(second["chat_history"]) == 4

    def test_separate_conversations_do_not_leak_history(self, client, pdf) -> None:
        document_id = upload(client, pdf).json()["document_id"]

        client.post("/api/v1/chat", json={"question": "Secret one", "document_id": document_id})
        other = client.post(
            "/api/v1/chat", json={"question": "Unrelated", "document_id": document_id}
        ).json()

        assert [m["content"] for m in other["chat_history"]] == ["Unrelated", other["answer"]]

    def test_reset_clears_conversation(self, client, pdf) -> None:
        document_id = upload(client, pdf).json()["document_id"]
        first = client.post("/api/v1/chat", json={"question": "A", "document_id": document_id})
        conversation_id = first.json()["conversation_id"]

        response = client.post(f"/api/v1/chat/reset?conversation_id={conversation_id}")

        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_top_k_is_validated(self, client, pdf) -> None:
        document_id = upload(client, pdf).json()["document_id"]
        response = client.post(
            "/api/v1/chat",
            json={"question": "hello", "document_id": document_id, "top_k": 99},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------- stats
class TestStats:
    def test_stats_reflect_ingested_documents(self, client, pdf) -> None:
        upload(client, pdf)

        body = client.get("/api/v1/stats").json()

        assert body["documents"] == 1
        assert body["ready_documents"] == 1
        assert body["total_pages"] == 2
        assert body["backend"] == "fake"
