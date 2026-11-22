"""Node 3: Vector similarity retriever over clinical guideline knowledge base.

Embeds the clinical query using the configured embedding model and executes
pgvector cosine distance queries against the rag.guideline_chunks table in PostgreSQL.
"""

import os

from openai import OpenAI
from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.chat.schemas import GraphState, RetrievedChunk
from app.core.config import settings
from app.model.guideline import GuidelineChunk


def _build_database_url() -> str:
    """Normalize database connection URL for synchronous SQLAlchemy engine."""
    db_url = os.getenv("DATABASE_URL") or settings.database_url
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    if "asyncpg" in db_url:
        db_url = db_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://", 1)
    if "ssl=require" in db_url:
        db_url = db_url.replace("ssl=require", "sslmode=require")
    return db_url


def _get_engine() -> Engine:
    """Create synchronous SQLAlchemy engine with connection pool pre-ping."""
    return create_engine(_build_database_url(), pool_pre_ping=True)


def _get_openai_client() -> tuple[OpenAI, str]:
    """Resolve OpenAI-compatible embedding client and target model."""
    api_key = os.getenv("OPENAI_API_KEY") or settings.arc_llm_key
    base_url = os.getenv("OPENAI_BASE_URL") or settings.arc_llm_endpoint
    client = OpenAI(api_key=api_key, base_url=base_url)
    model = os.getenv("EMBEDDING_MODEL") or settings.embedding_model
    return client, model


engine: Engine = _get_engine()


def retrieve_guidelines(state: GraphState) -> dict[str, object]:
    """Retrieve relevant clinical guideline chunks matching query semantics."""
    query = state.get("query", "")
    client, embedding_model = _get_openai_client()

    resp = client.embeddings.create(input=query, model=embedding_model)
    query_vector = resp.data[0].embedding

    with Session(engine) as session:
        stmt = (
            select(
                GuidelineChunk,
                (1 - GuidelineChunk.embedding.cosine_distance(query_vector)).label(
                    "similarity"
                ),
            )
            .where(GuidelineChunk.status == "active")
            .order_by(GuidelineChunk.embedding.cosine_distance(query_vector))
            .limit(4)
        )
        results = session.execute(stmt).all()

    # Threshold accommodates clinical semantic distance
    min_similarity = float(os.getenv("MIN_SIMILARITY_THRESHOLD", "0.55"))
    chunks: list[RetrievedChunk] = []
    for chunk, similarity in results:
        sim_val = float(similarity)
        if sim_val >= min_similarity:
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    guideline_title=chunk.guideline_title,
                    section_title=chunk.section_title,
                    content=chunk.content,
                    similarity=sim_val,
                )
            )

    return {"retrieved_chunks": chunks}
