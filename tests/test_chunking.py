"""Chunking behaviour."""

from __future__ import annotations

import pytest

from app.services.rag.chunking import RecursiveTextChunker
from app.services.rag.schemas import Page


def test_split_preserves_page_provenance() -> None:
    chunker = RecursiveTextChunker(chunk_size=100, chunk_overlap=10)
    pages = [Page(number=1, text="Alpha beta gamma."), Page(number=2, text="Delta epsilon.")]

    chunks = chunker.split(pages, document_id="doc_1")

    assert [chunk.page for chunk in chunks] == [1, 2]
    assert {chunk.document_id for chunk in chunks} == {"doc_1"}


def test_split_assigns_monotonic_chunk_index_per_page() -> None:
    chunker = RecursiveTextChunker(chunk_size=40, chunk_overlap=5)
    pages = [Page(number=1, text="word " * 200)]

    chunks = chunker.split(pages, document_id="doc_1")

    assert len(chunks) > 1
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_split_records_character_offsets() -> None:
    chunker = RecursiveTextChunker(chunk_size=50, chunk_overlap=5)
    text = "Retrieval augmented generation grounds answers in source documents."

    chunks = chunker.split([Page(number=1, text=text)], document_id="doc_1")

    assert chunks
    for chunk in chunks:
        assert chunk.char_start is not None
        assert chunk.char_end is not None
        assert text[chunk.char_start : chunk.char_end] == chunk.text


def test_split_skips_blank_pages() -> None:
    chunker = RecursiveTextChunker(chunk_size=100, chunk_overlap=10)
    pages = [Page(number=1, text="   \n  "), Page(number=2, text="Real content here.")]

    chunks = chunker.split(pages, document_id="doc_1")

    assert [chunk.page for chunk in chunks] == [2]


def test_overlap_preserves_boundary_text() -> None:
    chunker = RecursiveTextChunker(chunk_size=60, chunk_overlap=30)
    text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima"

    chunks = chunker.split([Page(number=1, text=text)], document_id="doc_1")

    assert len(chunks) >= 2
    first_tail = chunks[0].text.split()[-3:]
    assert any(word in chunks[1].text for word in first_tail)


def test_chunk_size_is_respected() -> None:
    chunker = RecursiveTextChunker(chunk_size=80, chunk_overlap=10)

    chunks = chunker.split(
        [Page(number=1, text="A first sentence. A second sentence. " + "filler " * 40)],
        document_id="doc_1",
    )

    assert all(len(chunk.text) <= 80 for chunk in chunks)


def test_ids_are_unique_and_prefixed() -> None:
    chunker = RecursiveTextChunker(chunk_size=30, chunk_overlap=5)
    pages = [Page(number=1, text="content " * 100), Page(number=2, text="more " * 100)]

    chunks = chunker.split(pages, document_id="doc_1")

    ids = [chunk.id for chunk in chunks]
    assert len(ids) == len(set(ids))
    assert all(chunk_id.startswith("chk_") for chunk_id in ids)


class TestValidation:
    @pytest.mark.parametrize(
        ("chunk_size", "chunk_overlap"),
        [(0, 0), (-1, 0), (100, 100), (100, 150)],
    )
    def test_rejects_invalid_geometry(self, chunk_size: int, chunk_overlap: int) -> None:
        with pytest.raises(ValueError):
            RecursiveTextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
