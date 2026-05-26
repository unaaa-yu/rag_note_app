"""
Central application configuration via Pydantic Settings.

All values come from environment variables (12-factor app).
Pydantic validates types at import time — misconfigured deployments
fail immediately with a clear error message.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings

# v1 simplification: every request is attributed to this user.
# Production should replace this with OAuth (e.g. Auth0, Supabase Auth)
# and extract the user_id from the JWT claim.
DEFAULT_USER_ID = "00000000-0000-0000-0000-000000000001"


class Settings(BaseSettings):
    # ── Database ──────────────────────────────────────────────────────────
    database_url: str

    # ── AI providers ──────────────────────────────────────────────────────
    openai_api_key: str
    anthropic_api_key: str

    # ── Embedding ─────────────────────────────────────────────────────────
    # DESIGN: 512-dim Matryoshka truncation of text-embedding-3-small.
    # See ARCHITECTURE.md §7 "Embedding Dimension Decision" for full rationale.
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 512

    # ── RAG retrieval ─────────────────────────────────────────────────────
    # Number of chunks returned by vector search and injected into the prompt.
    # Higher = more context, higher token cost, potential context dilution.
    top_k: int = 5

    # ── App ───────────────────────────────────────────────────────────────
    debug: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings singleton. Import this everywhere."""
    return Settings()


settings = get_settings()
