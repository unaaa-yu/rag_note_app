"""
FastAPI application factory.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.upload import router as upload_router
from app.api.chat import router as chat_router
from app.schemas.schemas import HealthOut


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Note App",
        description="PDF Q&A tool powered by pgvector + OpenAI embeddings + Claude",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],  # Vite dev server
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(upload_router)
    app.include_router(chat_router)

    @app.get("/health", response_model=HealthOut, tags=["meta"])
    def health() -> HealthOut:
        return HealthOut()

    return app


app = create_app()
