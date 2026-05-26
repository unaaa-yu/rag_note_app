"""
Chat / Q&A endpoints.

POST /ask                    — ask a question against one or more documents
GET  /conversations/{id}     — retrieve full conversation history
"""
from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.config import DEFAULT_USER_ID
from app.models.database import Conversation, Document, Message, get_db
from app.schemas.schemas import AskRequest, AskResponse, ConversationOut, MessageOut, SourceOut
from app.services.rag import answer_question

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, db: Session = Depends(get_db)) -> AskResponse:
    """
    Answer a question using RAG over the selected documents.

    Flow:
    1. Validate all document_ids belong to the default user and are "ready".
    2. Resolve or create a Conversation row.
    3. Persist the user Message.
    4. Call answer_question() → (answer_text, sources).
    5. Persist the assistant Message with sources as JSONB.
    6. Return AskResponse.
    """
    # 1. Validate documents
    for doc_id in body.document_ids:
        doc = db.get(Document, doc_id)
        if not doc or doc.user_id != DEFAULT_USER_ID:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found.")
        if doc.status != "ready":
            raise HTTPException(
                status_code=409,
                detail=f"Document '{doc.filename}' is not ready (status: {doc.status}).",
            )

    # 2. Resolve conversation
    if body.conversation_id:
        conv = db.get(Conversation, body.conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found.")
    else:
        conv = Conversation(
            id=str(uuid.uuid4()),
            user_id=DEFAULT_USER_ID,
            # If single doc, associate; multi-doc leaves document_id NULL
            document_id=body.document_ids[0] if len(body.document_ids) == 1 else None,
        )
        db.add(conv)
        db.flush()

    # 3. Persist user message
    user_msg = Message(
        id=str(uuid.uuid4()),
        conversation_id=conv.id,
        role="user",
        content=body.question,
        sources=[],
    )
    db.add(user_msg)
    db.flush()

    # 4. RAG — embed + retrieve + Claude
    try:
        answer_text, sources = answer_question(
            db=db,
            document_ids=body.document_ids,
            question=body.question,
        )
    except Exception as exc:
        logger.exception("RAG pipeline error: %s", exc)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {exc}")

    # 5. Persist assistant message
    sources_json = [
        {
            "chunk_id": s.chunk_id,
            "page_number": s.page_number,
            "content": s.content,
            "score": s.score,
        }
        for s in sources
    ]
    assistant_msg = Message(
        id=str(uuid.uuid4()),
        conversation_id=conv.id,
        role="assistant",
        content=answer_text,
        sources=sources_json,
    )
    db.add(assistant_msg)
    db.commit()

    db.refresh(assistant_msg)

    # 6. Build response
    source_outs = [
        SourceOut(
            chunk_id=s["chunk_id"],
            page_number=s["page_number"],
            content=s["content"],
            score=s["score"],
        )
        for s in sources_json
    ]
    msg_out = MessageOut(
        id=str(assistant_msg.id),
        role="assistant",
        content=answer_text,
        sources=source_outs,
        created_at=assistant_msg.created_at,
    )
    return AskResponse(conversation_id=str(conv.id), message=msg_out)


@router.get("/conversations/{conversation_id}", response_model=ConversationOut)
def get_conversation(conversation_id: str, db: Session = Depends(get_db)) -> ConversationOut:
    """Return the full message history for a conversation."""
    conv = db.get(Conversation, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .all()
    )

    message_outs = []
    for m in messages:
        sources = [
            SourceOut(
                chunk_id=s.get("chunk_id", ""),
                page_number=s.get("page_number", 0),
                content=s.get("content", ""),
                score=s.get("score", 0.0),
            )
            for s in (m.sources or [])
        ]
        message_outs.append(MessageOut(
            id=str(m.id),
            role=m.role,
            content=m.content,
            sources=sources,
            created_at=m.created_at,
        ))

    return ConversationOut(
        id=str(conv.id),
        document_id=str(conv.document_id) if conv.document_id else None,
        messages=message_outs,
        created_at=conv.created_at,
    )
