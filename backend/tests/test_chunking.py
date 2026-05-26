"""Tests for chunking.chunk_document()."""
import pytest

from app.services.pdf_parser import PageText
from app.services.chunking import chunk_document, TextChunk


SAMPLE_PAGES = [
    PageText(page_number=1, text="The quick brown fox jumps over the lazy dog. " * 30),
    PageText(page_number=2, text="Pack my box with five dozen liquor jugs. " * 30),
]


def test_returns_list_of_text_chunks():
    chunks = chunk_document(SAMPLE_PAGES, chunk_size=100, overlap=20)
    assert isinstance(chunks, list)
    assert all(isinstance(c, TextChunk) for c in chunks)


def test_chunk_index_sequential():
    chunks = chunk_document(SAMPLE_PAGES, chunk_size=100, overlap=20)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_overlap_ge_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_document(SAMPLE_PAGES, chunk_size=100, overlap=100)


def test_empty_pages_returns_empty():
    assert chunk_document([], chunk_size=100, overlap=20) == []


def test_page_number_recorded():
    chunks = chunk_document(SAMPLE_PAGES, chunk_size=100, overlap=20)
    valid_pages = {p.page_number for p in SAMPLE_PAGES}
    for chunk in chunks:
        assert chunk.page_number in valid_pages
