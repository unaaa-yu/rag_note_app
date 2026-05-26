"""
Prompt templates for the RAG answer-generation step.

DESIGN NOTES
────────────
The system prompt instructs Claude to:
- Answer only from the retrieved context (no hallucination).
- Cite page numbers when referencing specific passages.
- Admit uncertainty when the context doesn't cover the question.

TODO (step c) — implement build_rag_prompt():
    1. Format each SearchResult as:
           [Page {page_number}] {content}
    2. Join with double newlines.
    3. Return (system_prompt, user_message) tuple consumed by
       the Claude messages API.

Example skeleton:
    def build_rag_prompt(
        question: str,
        results: list[SearchResult],
    ) -> tuple[str, str]:
        context_blocks = [
            f"[Page {r.page_number}] {r.content}"
            for r in results
        ]
        context = "\n\n".join(context_blocks)
        system = SYSTEM_PROMPT
        user = USER_TEMPLATE.format(context=context, question=question)
        return system, user
"""
from __future__ import annotations

from app.services.vector_store import SearchResult

SYSTEM_PROMPT = """\
You are a helpful research assistant. Answer the user's question using ONLY
the document excerpts provided below. If the answer is not contained in the
excerpts, say "I don't have enough information in the document to answer that."

When you quote or reference specific information, cite the page number in
parentheses, e.g. (Page 3).

Be concise and precise. Do not add information beyond what the excerpts contain.
"""

USER_TEMPLATE = """\
Document excerpts:
{context}

Question: {question}
"""


def build_rag_prompt(
    question: str,
    results: list[SearchResult],
) -> tuple[str, str]:
    """
    Build the (system_prompt, user_message) pair for Claude.

    Args:
        question: The user's natural-language question.
        results:  Retrieved chunks from similarity_search(), ordered by score.

    Returns:
        (system_prompt, user_message) strings for the Claude messages API.

    """
    context_blocks = [
        f"[Page {r.page_number}] {r.content}"
        for r in results
    ]
    context = "\n\n".join(context_blocks)
    return SYSTEM_PROMPT, USER_TEMPLATE.format(context=context, question=question)
