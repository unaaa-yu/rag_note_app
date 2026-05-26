"""
SQLAlchemy ORM models.

Schema hierarchy:
    users ──< documents ──< chunks
                 └──< conversations ──< messages

Design notes:
- All PKs are UUID so rows can be created client-side without a round-trip.
- Chunk.embedding uses pgvector's Vector type (512-dim Matryoshka).
- Message.sources is JSONB: [{page, text, score}] written once, read with the row.
- Cascade deletes: removing a Document removes all its Chunks + Conversations + Messages.
"""
import uuid
from datetime import datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime, ForeignKey, Index, Integer,
    String, Text, create_engine,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column,
    relationship, sessionmaker,
)

from app.core.config import settings


class Base(DeclarativeBase):
    pass


# ── User ──────────────────────────────────────────────────────────────────────

class User(Base):
    """
    v1 simplification: only one row exists (DEFAULT_USER_ID from config).
    Schema is complete so auth can be layered on later without migrations.
    """
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # v1: email is nullable — demo user has no email.
    # Production: non-nullable + unique index.
    email: Mapped[Optional[str]] = mapped_column(String(255), unique=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    documents: Mapped[list["Document"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    conversations: Mapped[list["Conversation"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


# ── Document ──────────────────────────────────────────────────────────────────

class Document(Base):
    """
    One uploaded PDF.

    status lifecycle: "processing" → "ready" | "failed"
    BackgroundTask writes the final status after embedding completes.
    The frontend polls GET /documents/{id} until status != "processing".
    """
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    # "processing" | "ready" | "failed"
    status: Mapped[str] = mapped_column(String(20), default="processing", nullable=False)
    page_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    chunk_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # Stores traceback on failure — useful for debugging without grep-ing logs
    error_msg: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship(back_populates="documents")
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    conversations: Mapped[list["Conversation"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


# ── Chunk ─────────────────────────────────────────────────────────────────────

class Chunk(Base):
    """
    A text chunk extracted from a Document, with its 512-dim embedding.

    DESIGN: embedding is nullable so we can INSERT the row immediately after
    parsing (for progress tracking) and fill in the vector after the OpenAI
    call succeeds. In practice the pipeline does both atomically, but the
    nullable column makes partial failure recovery easier.

    The HNSW index is created by the Alembic migration AFTER the initial
    bulk insert — building on a populated table is significantly faster
    than incremental index maintenance.
    """
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    # 512-dim Matryoshka-truncated vector
    embedding: Mapped[Optional[list[float]]] = mapped_column(
        Vector(512), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    document: Mapped["Document"] = relationship(back_populates="chunks")

    # HNSW index is defined here as metadata for documentation purposes.
    # The actual CREATE INDEX is in the Alembic migration (not auto-generated
    # by SQLAlchemy) because pgvector indexes use non-standard syntax.
    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
    )


# ── Conversation ──────────────────────────────────────────────────────────────

class Conversation(Base):
    """
    A chat session tied to one or more documents.

    v1: tied to exactly one document (document_id FK).
    v2 upgrade path: add a join table document_conversations for multi-doc RAG.
    """
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    document: Mapped["Document"] = relationship(back_populates="conversations")
    user: Mapped["User"] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan",
        order_by="Message.created_at",
    )


# ── Message ───────────────────────────────────────────────────────────────────

class Message(Base):
    """
    One turn in a Conversation (either user question or assistant answer).

    sources: JSONB array present only on assistant messages.
    Schema: [{"page": 3, "text": "...", "score": 0.92}, ...]
    The frontend uses this to render page citation chips.
    """
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    # "user" | "assistant"
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # [{page: int, text: str, score: float}] — null for user messages
    sources: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")


# ── Engine + session factory ──────────────────────────────────────────────────

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db():
    """FastAPI dependency — yields a session and ensures it is closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
