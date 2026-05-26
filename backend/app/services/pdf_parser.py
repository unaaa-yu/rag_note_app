"""
PDF text extraction using PyMuPDF (fitz).

WHY PyMuPDF?
- 3–5× faster than pdfplumber / pdfminer on typical academic PDFs.
- Preserves page boundaries accurately — critical for citation page numbers.
- Handles multi-column layouts better than most alternatives.
- Pure Python install via pip; no system-level Poppler dependency.

LIMITATIONS (v1 scope):
- Text-layer PDFs only. Scanned image PDFs return empty pages.
  → v2: add OCR via pytesseract / Tesseract when page.get_text() is empty.
- Page numbers are structural (PDF page index), not printed numbers.
  → v2: attempt to parse printed page numbers from headers/footers.
"""
from __future__ import annotations

from dataclasses import dataclass

import fitz  # PyMuPDF


@dataclass
class PageText:
    """
    Text content of a single PDF page.

    page_number is 1-indexed to match printed page numbers in academic PDFs
    and to produce human-readable citations ("see page 3" not "see page 2").
    """
    page_number: int   # 1-indexed
    text: str


def parse_pdf(pdf_bytes: bytes) -> tuple[list[PageText], int]:
    """
    Extract text from PDF bytes, one PageText per non-empty page.

    Args:
        pdf_bytes: Raw bytes of the PDF file (from UploadFile.read()).

    Returns:
        (pages, total_page_count)
        pages:            List of PageText for every page with extractable text.
                          Empty pages (no text layer) are silently skipped.
        total_page_count: Total number of pages in the PDF including empty ones.
                          Stored on the Document row for display.

    Raises:
        ValueError: If the bytes are not a valid PDF.
        RuntimeError: If PyMuPDF cannot open the document.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Could not open PDF: {exc}") from exc

    total_pages = len(doc)
    pages: list[PageText] = []

    for i in range(total_pages):
        page = doc[i]
        # get_text("text") returns plain text; alternatives: "blocks", "html", "dict"
        text = page.get_text("text")
        cleaned = _clean_text(text)
        if cleaned:
            pages.append(PageText(page_number=i + 1, text=cleaned))

    doc.close()
    return pages, total_pages


def _clean_text(text: str) -> str:
    """
    Light cleanup of PyMuPDF output.

    - Collapse runs of blank lines to a single blank line.
    - Strip leading/trailing whitespace per line.
    - Drop lines that are only whitespace.

    We deliberately keep paragraph structure (double newlines) because
    the chunker uses it to prefer splitting at paragraph boundaries.
    """
    lines = text.splitlines()
    cleaned_lines: list[str] = []
    prev_blank = False

    for line in lines:
        stripped = line.strip()
        if stripped:
            cleaned_lines.append(stripped)
            prev_blank = False
        else:
            if not prev_blank and cleaned_lines:  # suppress leading/consecutive blanks
                cleaned_lines.append("")
            prev_blank = True

    return "\n".join(cleaned_lines).strip()
