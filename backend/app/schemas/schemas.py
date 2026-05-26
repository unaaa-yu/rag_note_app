"""
Pydantic request/response schemas.

Kept separate from SQLAlchemy models (app/models/database.py) so the API
contract is independent of the persistence layer.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ─── Document schemas ────────────────────────────────────────────────────────

class DocumentOut(BaseModel):
    """Returned after upload and in document listings."""
    id: str
    filename: str
    status: str                     # "processing" | "ready" | "failed"
    page_count: Optional[int]
    chunk_count: Optional[int]
    error_msg: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentListOut(BaseModel):
    documents: list[DocumentOut]


# ─── Chat / conversation schemas ─────────────────────────────────────────────

class AskRequest(BaseModel):
    """Body for POST /ask."""
    document_ids: list[str] = Field(
        ...,
        min_length=1,
        description="One or more document UUIDs to search against.",
    )
    question: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[str] = Field(
        None,
        description="Pass to continue an existing conversation; omit to start a new one.",
    )


class SourceOut(BaseModel):
    """A single retrieved chunk attached to an assistant answer."""
    chunk_id: str
    page_number: int
    content: str
    score: float


class MessageOut(BaseModel):
    """A single message in a conversation."""
    id: str
    role: str                       # "user" | "assistant"
    content: str
    sources: list[SourceOut] = []   # only populated for assistant messages
    created_at: datetime

    model_config = {"from_attributes": True}


class AskResponse(BaseModel):
    """Returned from POST /ask."""
    conversation_id: str
    message: MessageOut


class ConversationOut(BaseModel):
    """Full conversation history."""
    id: str
    document_id: Optional[str]
    messages: list[MessageOut]
    created_at: datetime

    model_config = {"from_attributes": True}


# ─── Health check ─────────────────────────────────────────────────────────────

class HealthOut(BaseModel):
    status: str = "ok"
