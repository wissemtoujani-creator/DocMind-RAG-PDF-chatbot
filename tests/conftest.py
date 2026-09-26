"""Shared fixtures.

The suite never loads a real embedding model or calls a real LLM. Fakes are
installed for both, so tests run in milliseconds and deterministically — the
network is only touched by tests explicitly marked ``integration``.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.repositories.conversations import InMemoryConversationRepository
from app.repositories.documents import InMemoryDocumentRepository
from app.services.rag.chunking import RecursiveTextChunker
from app.services.rag.pipeline import RAGPipeline
from app.services.rag.schemas import Chunk, RetrievedChunk

EMBED_DIM = 64


class FakeEmbedder:
    """Deterministic hashing embedder — stable across runs and processes."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        digest = hashlib.sha256(text.lower().encode("utf-8")).digest()
        raw = [digest[i % len(digest)] / 255.0 for i in range(EMBED_DIM)]
        norm = math.sqrt(sum(value * value for value in raw)) or 1.0
        return [value / norm for value in raw]


class FakeGenerator:
    """Records prompts and returns a canned answer."""

    def __init__(self, answer: str = "Stubbed answer [1].") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def generate(self, prompt: str):
        from app.services.rag.generator import Generation

        self.prompts.append(prompt)
        return Generation(text=self.answer, prompt_tokens=len(prompt) // 4, completion_tokens=8)


class InMemoryVectorStore:
    """Keyword-overlap retriever with cosine scoring, no external service."""

    backend = "fake"

    def __init__(self) -> None:
        self._chunks: dict[str, list[Chunk]] = {}
        self._vectors: dict[str, list[float]] = {}

    def add_chunks(self, chunks: list[Chunk], *, document_id: str) -> int:
        embedder = FakeEmbedder()
        self._chunks.setdefault(document_id, []).extend(chunks)
        for chunk in chunks:
            self._vectors[chunk.id] = embedder.embed_query(chunk.text)
        return len(chunks)

    def similarity_search(
        self, query: str, *, document_id: str, k: int
    ) -> list[RetrievedChunk]:
        query_vector = FakeEmbedder().embed_query(query)
        scored: list[RetrievedChunk] = []
        for chunk in self._chunks.get(document_id, []):
            scored.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=round(_cosine(query_vector, self._vectors[chunk.id]), 6),
                )
            )
        scored.sort(key=lambda item: item.score or 0.0, reverse=True)
        return scored[:k]

    def delete_document(self, document_id: str) -> int:
        chunks = self._chunks.pop(document_id, [])
        for chunk in chunks:
            self._vectors.pop(chunk.id, None)
        return len(chunks)

    def count(self, document_id: str | None = None) -> int:
        if document_id is None:
            return sum(len(items) for items in self._chunks.values())
        return len(self._chunks.get(document_id, []))

    def health(self) -> bool:
        return True


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        environment="test",
        log_level="CRITICAL",
        log_format="json",
        storage_dir=tmp_path / "storage",
        chroma_persist_dir=tmp_path / "chroma",
        groq_api_key=None,
    )


@pytest.fixture
def generator() -> FakeGenerator:
    return FakeGenerator()


@pytest.fixture
def pipeline(settings: Settings, generator: FakeGenerator) -> RAGPipeline:
    return RAGPipeline(
        vector_store=InMemoryVectorStore(),
        embedder=FakeEmbedder(),
        generator=generator,
        chunker=RecursiveTextChunker(chunk_size=200, chunk_overlap=20),
        top_k=3,
        preview_chars=50,
    )


@pytest.fixture
def documents() -> InMemoryDocumentRepository:
    return InMemoryDocumentRepository()


@pytest.fixture
def conversations() -> InMemoryConversationRepository:
    return InMemoryConversationRepository()


@pytest.fixture
def client(
    settings: Settings,
    pipeline: RAGPipeline,
    documents: InMemoryDocumentRepository,
    conversations: InMemoryConversationRepository,
) -> Iterator[TestClient]:
    """A TestClient wired to fakes, exercising the real app factory."""
    from app.main import create_app

    app = create_app(settings)
    # Replace lifespan-built collaborators with the fakes above.
    app.router.lifespan_context = _noop_lifespan(
        app, pipeline, documents, conversations
    )
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def _noop_lifespan(app: Any, *collaborators: Any):
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_app: Any):
        app.state.pipeline, app.state.documents, app.state.conversations = collaborators
        yield

    return lifespan


# --------------------------------------------------------------------- PDFs
def make_pdf(path: Path, pages: list[str]) -> Path:
    """Write a minimal but valid multi-page PDF without external fixtures.

    Object numbering: 1 = catalog, 2 = page tree, 3 = font, then a
    (page, content) pair per page starting at 4.
    """
    objects: list[bytes] = []
    page_count = len(pages)
    font_obj = 3

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = b" ".join(f"{4 + i * 2} 0 R".encode() for i in range(page_count))
    objects.append(
        b"<< /Type /Pages /Count "
        + str(page_count).encode()
        + b" /Kids ["
        + kids
        + b"] >>"
    )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for index, text in enumerate(pages):
        page_obj = 4 + index * 2
        content_obj = page_obj + 1
        stream = _content_stream(text)
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_obj} 0 R >> >> "
            f"/Contents {content_obj} 0 R >>".encode()
        )
        objects.append(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n"
        f"{xref_offset}\n%%EOF\n"
    ).encode()

    path.write_bytes(bytes(out))
    return path


def _content_stream(text: str) -> bytes:
    escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
    body = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1", errors="replace")
    return b"q\n" + body + b"\nQ"
