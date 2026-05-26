"""
pgvector persistence layer.

Responsibilities:
- Bulk-insert TextChunks with their embeddings into the `chunks` table.
- Run cosine-similarity search against the HNSW index for retrieval.

DESIGN: raw SQL for both operations.

For INSERT, raw SQL via executemany() is 10–50× faster than
ORM .add() in a loop because it avoids per-row Python overhead and
sends one round-trip (with executemany or COPY).

For SELECT, the pgvector <=> operator (cosine distance) requires syntax
that the SQLAlchemy ORM cannot express without pgvector's custom dialect
extension. Raw SQL is simpler and more transparent.

COSINE DISTANCE vs COSINE SIMILARITY
──────────────────────────────────────
pgvector's <=> operator returns DISTANCE (0 = identical, 2 = opposite).
We expose SIMILARITY (score = 1 − distance) so higher is better,
which is the convention users and the frontend expect.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.services.chunking import TextChunk

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A retrieved chunk with its similarity score."""
    chunk_id: str
    content: str
    page_number: int
    chunk_index: int
    score: float          # cosine similarity in [0, 1]; higher = more relevant


def insert_chunks(
    db: Session,
    document_id: str,
    chunks: list[TextChunk],
    embeddings: list[list[float]],
) -> None:
    """
    Bulk-insert all chunks for a document in a single transaction.

    Args:
        db:           SQLAlchemy session (owned by caller).
        document_id:  UUID string of the parent Document row.
        chunks:       TextChunk objects from chunking.chunk_document().
        embeddings:   Parallel list of embedding vectors from Embedder.embed_texts().
                      Must satisfy len(embeddings) == len(chunks).

    DESIGN: we use raw executemany with a parameterized INSERT rather than
    the ORM to avoid one Python object allocation per row.
    The vector is passed as a Postgres literal string '[0.1, 0.2, ...]'
    which pgvector's input parser accepts.
    """
    if not chunks:
        return

    assert len(chunks) == len(embeddings), (
        f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must have the same length"
    )

    rows = [
        {
            "id": str(uuid.uuid4()),
            "document_id": document_id,
            "content": chunk.content,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            # pgvector accepts '[f1,f2,...]' string literal
            "embedding": "[" + ",".join(map(str, vec)) + "]",
        }
        for chunk, vec in zip(chunks, embeddings)
    ]

    db.execute(
        text("""
            INSERT INTO chunks (id, document_id, content, page_number, chunk_index, embedding)
            VALUES (
                :id ::uuid,
                :document_id ::uuid,
                :content,
                :page_number,
                :chunk_index,
                :embedding ::vector
            )
        """),
        rows,
    )
    db.commit()

    logger.info(
        "Inserted %d chunks for document %s", len(chunks), document_id
    )


def similarity_search(
    db: Session,
    document_id: str,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[SearchResult]:
    """
    Return the top-k most similar chunks for a query vector.

    Uses the HNSW index built by the Alembic migration (ix_chunks_embedding_hnsw).
    Filters by document_id so results are scoped to the selected document(s).

    Args:
        db:              SQLAlchemy session.
        document_id:     UUID string — only chunks from this document are searched.
        query_embedding: 512-dim vector from Embedder.embed_query().
        top_k:           Maximum number of results to return.

    Returns:
        List of SearchResult ordered by score descending (most relevant first).

    QUERY PLAN NOTE:
    pgvector uses the HNSW index when ORDER BY uses <=> and there's a LIMIT.
    Adding WHERE document_id = ... forces a filter on top of the index scan.
    For large corpora with many documents, consider a partition-per-document
    strategy or a composite index. For a portfolio project this is fine.
    """
    vec_literal = "[" + ",".join(map(str, query_embedding)) + "]"

    rows = db.execute(
        text("""
            SELECT
                id,
                content,
                page_number,
                chunk_index,
                1 - (embedding <=> :vec ::vector) AS score
            FROM chunks
            WHERE document_id = :doc_id ::uuid
            ORDER BY embedding <=> :vec ::vector
            LIMIT :k
        """),
        {"vec": vec_literal, "doc_id": str(document_id), "k": top_k},
    ).fetchall()

    return [
        SearchResult(
            chunk_id=str(row.id),
            content=row.content,
            page_number=row.page_number,
            chunk_index=row.chunk_index,
            score=round(float(row.score), 4),
        )
        for row in rows
    ]


def similarity_search_multi(
    db: Session,
    document_ids: list[str],
    query_embedding: list[float],
    top_k: int = 5,
) -> list[SearchResult]:
    """
    Search across multiple documents (multi-doc RAG).

    Used when the user has selected several documents in the sidebar.
    Returns the global top-k results across all selected documents,
    re-ranked by score.

    DESIGN: we pass the list as a Postgres array literal to avoid N queries.
    """
    if not document_ids:
        return []

    vec_literal = "[" + ",".join(map(str, query_embedding)) + "]"
    # Build Postgres UUID array literal: '{"uuid1","uuid2"}'
    ids_literal = "{" + ",".join(f'"{d}"' for d in document_ids) + "}"

    rows = db.execute(
        text("""
            SELECT
                id,
                content,
                page_number,
                chunk_index,
                1 - (embedding <=> :vec ::vector) AS score
            FROM chunks
            WHERE document_id = ANY(:ids ::uuid[])
            ORDER BY embedding <=> :vec ::vector
            LIMIT :k
        """),
        {"vec": vec_literal, "ids": ids_literal, "k": top_k},
    ).fetchall()

    return [
        SearchResult(
            chunk_id=str(row.id),
            content=row.content,
            page_number=row.page_number,
            chunk_index=row.chunk_index,
            score=round(float(row.score), 4),
        )
        for row in rows
    ]
