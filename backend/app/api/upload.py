"""
Document upload and management endpoints.

Routes
──────
POST /documents/upload     — upload a PDF; kicks off background processing
GET  /documents            — list all documents for the default user
GET  /documents/{id}       — get a single document (for status polling)
DELETE /documents/{id}     — delete document + all chunks (CASCADE)

Background pipeline (process_document)
────────────────────────────────────────
1. parse_pdf()       — extract text per page via PyMuPDF
2. chunk_document()  — sliding-window tokenizer via tiktoken
3. Embedder.embed_texts() — batch embed with OpenAI
4. insert_chunks()   — bulk INSERT into pgvector

Status transitions: "processing" → "ready" | "failed"
The status and chunk_count are updated in place on the Document row.

DESIGN: FastAPI BackgroundTasks keeps this simple (no Celery/Redis).
Tradeoff: if the server restarts mid-processing, the document stays
"processing" forever. For a portfolio project this is acceptable.
A production system would use a durable queue and a periodic watchdog
to reset stale "processing" rows.
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from app.core.config import DEFAULT_USER_ID
from app.models.database import Document, Chunk, get_db
from app.schemas.schemas import DocumentOut, DocumentListOut
from app.services.pdf_parser import parse_pdf
from app.services.chunking import chunk_document
from app.services.embedder import Embedder
from app.services.vector_store import insert_chunks

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

_embedder = Embedder()


# ─── Background task ──────────────────────────────────────────────────────────

def process_document(document_id: str, pdf_bytes: bytes, db: Session) -> None:
    """
    Full ingestion pipeline run in the background after upload.

    Updates Document.status to "ready" on success or "failed" on error.
    Always commits so the polling endpoint sees the final state.
    """
    try:
        # 1. Parse PDF
        pages, total_pages = parse_pdf(pdf_bytes)
        if not pages:
            raise ValueError("PDF contains no extractable text (scanned image?)")

        # 2. Chunk
        chunks = chunk_document(pages)

        # 3. Embed
        texts = [c.content for c in chunks]
        embeddings = _embedder.embed_texts(texts)

        # 4. Insert into pgvector
        insert_chunks(db, document_id, chunks, embeddings)

        # 5. Mark ready
        doc = db.get(Document, document_id)
        if doc:
            doc.status = "ready"
            doc.page_count = total_pages
            doc.chunk_count = len(chunks)
            db.commit()

        logger.info(
            "Document %s processed: %d pages, %d chunks",
            document_id, total_pages, len(chunks),
        )

    except Exception as exc:
        logger.exception("Failed to process document %s: %s", document_id, exc)
        doc = db.get(Document, document_id)
        if doc:
            doc.status = "failed"
            doc.error_msg = str(exc)[:500]
            db.commit()


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=DocumentOut, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> DocumentOut:
    """
    Accept a PDF upload, persist a Document row, and enqueue processing.

    Returns 202 Accepted immediately with status="processing".
    The client should poll GET /documents/{id} until status is "ready" or "failed".
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()
    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    doc = Document(
        id=str(uuid.uuid4()),
        user_id=DEFAULT_USER_ID,
        filename=file.filename,
        status="processing",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    background_tasks.add_task(process_document, doc.id, pdf_bytes, db)
    logger.info("Queued document %s (%s) for processing", doc.id, file.filename)

    return DocumentOut.model_validate(doc)


@router.get("", response_model=DocumentListOut)
def list_documents(db: Session = Depends(get_db)) -> DocumentListOut:
    """Return all documents for the default user, newest first."""
    docs = (
        db.query(Document)
        .filter(Document.user_id == DEFAULT_USER_ID)
        .order_by(Document.created_at.desc())
        .all()
    )
    return DocumentListOut(documents=[DocumentOut.model_validate(d) for d in docs])


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: str, db: Session = Depends(get_db)) -> DocumentOut:
    """Get a single document by ID (used by the frontend to poll processing status)."""
    doc = db.get(Document, document_id)
    if not doc or doc.user_id != DEFAULT_USER_ID:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DocumentOut.model_validate(doc)


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: str, db: Session = Depends(get_db)) -> None:
    """
    Delete a document and all its chunks (CASCADE handles chunk deletion).
    Also deletes all conversations referencing this document.
    """
    doc = db.get(Document, document_id)
    if not doc or doc.user_id != DEFAULT_USER_ID:
        raise HTTPException(status_code=404, detail="Document not found.")
    db.delete(doc)
    db.commit()
