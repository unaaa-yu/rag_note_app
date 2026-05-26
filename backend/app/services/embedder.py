"""
OpenAI embedding client.

DESIGN: wraps the OpenAI client so the rest of the codebase never imports
openai directly. If we switch providers (e.g. Cohere, local model via Ollama),
only this file changes.

MATRYOSHKA DIMENSIONS
─────────────────────
text-embedding-3-small supports Matryoshka Representation Learning: passing
`dimensions=512` truncates the 1536-dim output to 512 dimensions.
The model was trained to preserve information in the leading dimensions,
so quality degrades gracefully (~2–4% recall loss at 512 vs 1536).
See ARCHITECTURE.md §7 for full rationale.

BATCHING
────────
OpenAI accepts up to 2048 inputs per request but recommends ≤ 100 for
stability. We batch at BATCH_SIZE=100 and collect results in order.
The API guarantees response order matches input order.
"""
from __future__ import annotations

import logging
from typing import Optional

from openai import OpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

# Stay well under OpenAI's 2048-input limit per request
BATCH_SIZE = 100


class Embedder:
    """
    Thin wrapper around OpenAI's embeddings endpoint.

    Usage:
        embedder = Embedder()
        vectors = embedder.embed_texts(["chunk one", "chunk two"])
        query_vec = embedder.embed_query("what is the main contribution?")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
    ) -> None:
        self._client = OpenAI(api_key=api_key or settings.openai_api_key)
        self._model = model or settings.embedding_model
        self._dimensions = dimensions or settings.embedding_dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of text strings, returning one vector per string.

        Processes in batches of BATCH_SIZE to stay within API limits.
        Strips empty strings before sending (OpenAI rejects them).

        Args:
            texts: List of non-empty strings to embed.

        Returns:
            List of float vectors in the same order as `texts`.
            Length = len(texts), each vector has self._dimensions floats.

        Raises:
            openai.APIError: on authentication failure or rate limits.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for batch_start in range(0, len(texts), BATCH_SIZE):
            batch = texts[batch_start : batch_start + BATCH_SIZE]
            # Sanitize: replace empty strings with a space (API rejects "")
            batch = [t if t.strip() else " " for t in batch]

            logger.debug(
                "Embedding batch %d–%d of %d",
                batch_start + 1,
                min(batch_start + BATCH_SIZE, len(texts)),
                len(texts),
            )

            response = self._client.embeddings.create(
                model=self._model,
                input=batch,
                dimensions=self._dimensions,   # Matryoshka truncation
            )
            # API guarantees same order as input
            all_embeddings.extend(item.embedding for item in response.data)

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string.

        Convenience wrapper — identical to embed_texts([text])[0] but
        semantically clearer at call sites.
        """
        return self.embed_texts([text])[0]
