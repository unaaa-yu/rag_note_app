"""
Unit tests for pdf_parser.parse_pdf().

These tests use a tiny in-memory PDF built with PyMuPDF so no fixture files
are needed. PyMuPDF is already in requirements.txt.
"""
import pytest
import fitz

from app.services.pdf_parser import parse_pdf, PageText


def _make_pdf(pages: list[str]) -> bytes:
    """Create a minimal in-memory PDF with one text block per page."""
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    return doc.tobytes()


def test_single_page():
    pdf = _make_pdf(["Hello from page one."])
    result, total = parse_pdf(pdf)

    assert total == 1
    assert len(result) == 1
    assert result[0].page_number == 1
    assert "Hello from page one" in result[0].text


def test_multi_page_numbering():
    pdf = _make_pdf(["Page A", "Page B", "Page C"])
    result, total = parse_pdf(pdf)

    assert total == 3
    assert [p.page_number for p in result] == [1, 2, 3]


def test_empty_pages_skipped():
    # Build a 3-page PDF where page 2 has no text
    doc = fitz.open()
    doc.new_page()  # page 1 — no text
    p2 = doc.new_page()
    p2.insert_text((72, 72), "Only page with text")
    doc.new_page()  # page 3 — no text
    pdf = doc.tobytes()

    result, total = parse_pdf(pdf)

    assert total == 3
    assert len(result) == 1
    assert result[0].page_number == 2


def test_invalid_bytes_raises():
    with pytest.raises(ValueError, match="Could not open PDF"):
        parse_pdf(b"not a pdf")
