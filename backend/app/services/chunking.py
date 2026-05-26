"""
Text chunking strategy for RAG ingestion.

WHY CHUNKING MATTERS IN A RAG PIPELINE
───────────────────────────────────────
The embedding model (text-embedding-3-small) has a hard limit of 8191 tokens.
More importantly, embedding quality degrades when a single chunk is too long:
the vector becomes a "soup" of many topics, making it harder for cosine
similarity to match a narrow question to the right passage.

Conversely, chunks that are too short lack the context needed for the LLM
to form a coherent answer (a 2-sentence snippet is often ambiguous).

The overlap parameter solves the boundary problem: if a key sentence falls
at the boundary between chunk N and chunk N+1, at least one of them will
contain it fully, so retrieval won't miss it.

RECOMMENDED PARAMETERS FOR v1
──────────────────────────────
chunk_size   = 500 tokens  ≈ 375 words ≈ 1–2 dense paragraphs
chunk_overlap = 100 tokens  = 20% overlap

These are reasonable defaults for academic papers and technical documents.
Tune by measuring RAGAS retrieval recall on your own corpus.

ALGORITHM (sliding window with token budget)
────────────────────────────────────────────
1. Concatenate all page text, tracking page-boundary positions.
2. Tokenize the full text with tiktoken (cl100k_base).
3. Slide a window of `chunk_size` tokens, advancing by `chunk_size - overlap`.
4. Decode each window back to text.
5. Record which page the window starts on (for citation page numbers).
6. Return list[TextChunk].

PACKAGE REQUIRED
────────────────
pip install tiktoken
"""
from __future__ import annotations

from dataclasses import dataclass

from app.services.pdf_parser import PageText


@dataclass
class TextChunk:
    """
    A chunk of text ready for embedding.

    content:     The actual text to embed and store.
    page_number: Page where this chunk starts (1-indexed, for citations).
    chunk_index: Position in the document-level sequence (0-indexed).
    """
    content: str
    page_number: int
    chunk_index: int


def chunk_document(
    pages: list[PageText],
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[TextChunk]:
    """
    Split a list of PageText objects into overlapping TextChunks.

    Args:
        pages:      Output of pdf_parser.parse_pdf() — one item per page.
        chunk_size: Maximum tokens per chunk (default 500).
        overlap:    Token overlap between consecutive chunks (default 100).
                    Must be < chunk_size.

    Returns:
        List of TextChunk objects, ordered by position in the document.
        Empty if pages is empty.

    Raises:
        NotImplementedError: Until you implement this function.
        ValueError: If overlap >= chunk_size.

    ──────────────────────────────────────────────────────────────────────
    TODO — implement this function. Step-by-step guide:

    Step 1: Validate inputs.
        if overlap >= chunk_size:
            raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")
        if not pages:
            return []

    Step 2: Build a flat token list with a page-boundary map.
        ```python
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")

        all_tokens: list[int] = []
        # Maps token index → page_number (only at page-start boundaries)
        token_to_page: dict[int, int] = {}

        for page in pages:
            token_to_page[len(all_tokens)] = page.page_number
            all_tokens.extend(enc.encode(page.text))
        ```

    Step 3: Resolve token index → page number.
        ```python
        def get_page_at(token_idx: int) -> int:
            # Walk backwards from token_idx to find the nearest recorded page start.
            page = pages[0].page_number  # fallback
            for boundary, pg in token_to_page.items():
                if boundary <= token_idx:
                    page = pg
            return page
        ```
        Tip: pre-sort token_to_page items once for O(n) lookup.

    Step 4: Slide the window.
        ```python
        step = chunk_size - overlap
        chunks: list[TextChunk] = []
        chunk_index = 0

        for start in range(0, len(all_tokens), step):
            end = min(start + chunk_size, len(all_tokens))
            window_tokens = all_tokens[start:end]

            content = enc.decode(window_tokens).strip()
            if not content:
                continue

            page_number = get_page_at(start)
            chunks.append(TextChunk(
                content=content,
                page_number=page_number,
                chunk_index=chunk_index,
            ))
            chunk_index += 1

            if end == len(all_tokens):
                break   # avoid an empty final window
        ```

    Step 5: Return chunks.
        return chunks

    RESOURCES:
    - tiktoken docs: https://github.com/openai/tiktoken
    - Chunking strategies: https://www.pinecone.io/learn/chunking-strategies/
    - RAGAS for measuring retrieval quality: https://docs.ragas.io
    ──────────────────────────────────────────────────────────────────────
    """
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")
    if not pages:
        return []

    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    all_tokens: list[int] = []
    token_to_page: dict[int, int] = {}

    for page in pages:
        token_to_page[len(all_tokens)] = page.page_number
        all_tokens.extend(enc.encode(page.text))

    # Pre-sort boundaries once for O(n) lookup per chunk
    sorted_boundaries = sorted(token_to_page.items())

    def get_page_at(token_idx: int) -> int:
        page = pages[0].page_number
        for boundary, pg in sorted_boundaries:
            if boundary <= token_idx:
                page = pg
            else:
                break
        return page

    step = chunk_size - overlap
    chunks: list[TextChunk] = []
    chunk_index = 0

    for start in range(0, len(all_tokens), step):
        end = min(start + chunk_size, len(all_tokens))
        window_tokens = all_tokens[start:end]

        content = enc.decode(window_tokens).strip()
        if not content:
            continue

        chunks.append(TextChunk(
            content=content,
            page_number=get_page_at(start),
            chunk_index=chunk_index,
        ))
        chunk_index += 1

        if end == len(all_tokens):
            break

    return chunks
