"""Pipeline orchestration: ingestion, retrieval, citation, and deletion."""

from __future__ import annotations

import pytest
from tests.conftest import FakeEmbedder, FakeGenerator, InMemoryVectorStore, make_pdf

from app.core.errors import PayloadTooLargeError, UnsupportedFileTypeError, ValidationError
from app.services.rag.chunking import RecursiveTextChunker
from app.services.rag.generator import Generation
from app.services.rag.loader import extract_pages, looks_like_pdf, validate_upload
from app.services.rag.pipeline import NO_CONTEXT_ANSWER
from app.services.rag.schemas import ChatMessage


@pytest.fixture
def sample_pdf(tmp_path):
    return make_pdf(
        tmp_path / "handbook.pdf",
        pages=[
            "The refund window is thirty days from the invoice date.",
            "Support is available from nine to five on weekdays.",
        ],
    )


def build_pipeline(*, answer: str = "Stubbed [1].") -> tuple:
    store = InMemoryVectorStore()
    generator = FakeGenerator(answer)
    from app.services.rag.pipeline import RAGPipeline

    pipeline = RAGPipeline(
        vector_store=store,
        embedder=FakeEmbedder(),
        generator=generator,
        chunker=RecursiveTextChunker(chunk_size=200, chunk_overlap=20),
        top_k=2,
        preview_chars=20,
    )
    return pipeline, store, generator


# ------------------------------------------------------------------- loading
class TestValidation:
    def test_rejects_non_pdf_extension(self) -> None:
        with pytest.raises(UnsupportedFileTypeError):
            validate_upload("notes.txt", 100, 1_000_000)

    def test_rejects_empty_file(self) -> None:
        with pytest.raises(ValidationError):
            validate_upload("a.pdf", 0, 1_000_000)

    def test_rejects_oversized_file(self) -> None:
        with pytest.raises(PayloadTooLargeError):
            validate_upload("a.pdf", 2_000, 1_000)

    def test_strips_directory_from_filename(self) -> None:
        assert validate_upload("../../etc/passwd.pdf", 10, 1_000_000) == "passwd.pdf"

    def test_magic_byte_detection(self, sample_pdf) -> None:
        assert looks_like_pdf(sample_pdf.read_bytes()[:1024])
        assert not looks_like_pdf(b"PK\x03\x04zip")


class TestExtraction:
    def test_extracts_page_count(self, sample_pdf) -> None:
        assert len(extract_pages(sample_pdf)) == 2

    def test_rejects_non_pdf_bytes(self, tmp_path) -> None:
        broken = tmp_path / "broken.pdf"
        broken.write_bytes(b"not a pdf at all")
        with pytest.raises(ValidationError):
            extract_pages(broken)


# ------------------------------------------------------------------ ingestion
class TestIngest:
    def test_ingest_populates_record_and_store(self, sample_pdf) -> None:
        pipeline, store, _ = build_pipeline()

        result = pipeline.ingest(
            pdf_path=sample_pdf,
            document_id="doc_1",
            filename="handbook.pdf",
            size_bytes=sample_pdf.stat().st_size,
        )

        assert result.document.pages == 2
        assert result.document.chunks > 0
        assert store.count("doc_1") == result.document.chunks
        assert result.document.is_ready

    def test_ingest_is_idempotent_per_document(self, sample_pdf) -> None:
        pipeline, store, _ = build_pipeline()
        for _ in range(2):
            pipeline.ingest(
                pdf_path=sample_pdf,
                document_id="doc_1",
                filename="handbook.pdf",
                size_bytes=0,
            )
        # The second pass appends; it must not corrupt document scoping.
        assert store.count("doc_1") > 0

    def test_delete_removes_only_that_document(self, sample_pdf) -> None:
        pipeline, store, _ = build_pipeline()
        for document_id in ("doc_1", "doc_2"):
            pipeline.ingest(
                pdf_path=sample_pdf,
                document_id=document_id,
                filename="handbook.pdf",
                size_bytes=0,
            )

        removed = pipeline.delete("doc_1")

        assert removed > 0
        assert store.count("doc_1") == 0
        assert store.count("doc_2") > 0


# ------------------------------------------------------------------ retrieval
class TestRetrieve:
    def test_retrieval_is_scoped_to_document(self, sample_pdf) -> None:
        pipeline, _, _ = build_pipeline()
        pipeline.ingest(pdf_path=sample_pdf, document_id="doc_1", filename="a.pdf", size_bytes=0)
        pipeline.ingest(pdf_path=sample_pdf, document_id="doc_2", filename="b.pdf", size_bytes=0)

        assert pipeline.retrieve("refund", document_id="doc_1", top_k=5)
        assert pipeline.retrieve("refund", document_id="doc_2", top_k=5)

    def test_top_k_is_respected(self, sample_pdf) -> None:
        pipeline, _, _ = build_pipeline()
        pipeline.ingest(pdf_path=sample_pdf, document_id="doc_1", filename="a.pdf", size_bytes=0)

        assert len(pipeline.retrieve("refund window", document_id="doc_1", top_k=1)) == 1


# ------------------------------------------------------------------ answering
class TestAnswer:
    def test_answer_includes_numbered_citations(self, sample_pdf) -> None:
        pipeline, _, _ = build_pipeline(answer="The window is thirty days [1].")
        pipeline.ingest(pdf_path=sample_pdf, document_id="doc_1", filename="a.pdf", size_bytes=0)

        result = pipeline.answer(
            "What is the refund window?",
            document_id="doc_1",
            filename="handbook.pdf",
        )

        assert result.citations
        assert [c.marker for c in result.citations] == list(
            range(1, len(result.citations) + 1)
        )
        assert all(c.source == "handbook.pdf" for c in result.citations)
        assert result.chunks_used == len(result.citations)
        assert not result.degraded

    def test_citation_preview_is_truncated(self, sample_pdf) -> None:
        pipeline, _, _ = build_pipeline()
        pipeline.ingest(pdf_path=sample_pdf, document_id="doc_1", filename="a.pdf", size_bytes=0)

        result = pipeline.answer("refund", document_id="doc_1", filename="a.pdf")

        assert all(len(c.preview) <= 21 for c in result.citations)  # 20 chars + ellipsis

    def test_no_context_triggers_refusal_not_error(self) -> None:
        pipeline, _, generator = build_pipeline()

        result = pipeline.answer("anything", document_id="missing", filename="a.pdf")

        assert result.answer == NO_CONTEXT_ANSWER
        assert result.degraded
        assert result.citations == []
        assert generator.prompts == []

    def test_history_is_rendered_into_the_prompt(self, sample_pdf) -> None:
        pipeline, _, generator = build_pipeline()
        pipeline.ingest(pdf_path=sample_pdf, document_id="doc_1", filename="a.pdf", size_bytes=0)

        pipeline.answer(
            "And on weekends?",
            document_id="doc_1",
            filename="a.pdf",
            history=[ChatMessage(role="user", content="When is support open?")],
        )

        assert "Conversation so far" in generator.prompts[-1]
        assert "When is support open?" in generator.prompts[-1]

    def test_generator_failure_propagates(self, sample_pdf) -> None:
        from app.core.errors import ProviderError

        pipeline, _, generator = build_pipeline()
        pipeline.ingest(pdf_path=sample_pdf, document_id="doc_1", filename="a.pdf", size_bytes=0)

        def explode(_prompt: str) -> Generation:
            raise ProviderError("upstream is down")

        generator.generate = explode  # type: ignore[method-assign]

        with pytest.raises(ProviderError):
            pipeline.answer("refund", document_id="doc_1", filename="a.pdf")

    def test_empty_generation_falls_back_to_refusal(self, sample_pdf) -> None:
        pipeline, _, _ = build_pipeline(answer="")
        pipeline.ingest(pdf_path=sample_pdf, document_id="doc_1", filename="a.pdf", size_bytes=0)

        result = pipeline.answer("refund", document_id="doc_1", filename="a.pdf")

        assert result.answer == NO_CONTEXT_ANSWER

    def test_health_reports_backend(self) -> None:
        pipeline, _, _ = build_pipeline()
        assert pipeline.backend == "fake"
        assert pipeline.health() is True
