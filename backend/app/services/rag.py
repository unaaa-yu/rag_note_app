"""
RAG answer-generation service.

Wires together: vector search → prompt building → Claude API call.
Called by POST /ask.
"""
from __future__ import annotations

import logging

import anthropic

from app.core.config import settings
from app.services.embedder import Embedder
from app.services.vector_store import similarity_search_multi, SearchResult
from app.services.prompt import build_rag_prompt
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

_embedder = Embedder()
_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

CLAUDE_MODEL = "claude-sonnet-4-5"
MAX_TOKENS = 1024


def answer_question(
    db: Session,
    document_ids: list[str],
    question: str,
) -> tuple[str, list[SearchResult]]:
    """
    Retrieve relevant chunks and generate an answer with Claude.

    Args:
        db:           SQLAlchemy session.
        document_ids: UUIDs of documents to search.
        question:     User's natural-language question.

    Returns:
        (answer_text, sources) — sources are the retrieved chunks used as context.
    """
    # 1. Embed the question
    query_vec = _embedder.embed_query(question)

    # 2. Retrieve top-k chunks
    results = similarity_search_multi(
        db=db,
        document_ids=document_ids,
        query_embedding=query_vec,
        top_k=settings.top_k,
    )

    if not results:
        return (
            "I couldn't find any relevant content in the selected document(s) to answer your question.",
            [],
        )

    # 3. Build prompt
    system_prompt, user_message = build_rag_prompt(question, results)

    # 4. Call Claude
    logger.debug("Calling Claude with %d context chunks", len(results))
    response = _client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    answer = response.content[0].text
    return answer, results
