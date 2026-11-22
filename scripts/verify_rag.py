"""Step 9: RAG Retrieval Verification Script.

Proves that the database knowledge base, pgvector HNSW index,
embeddings, and SQLAlchemy queries are healthy before starting Phase 2.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import NamedTuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai import OpenAI
from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings
from app.core.db import SessionLocal, engine
from app.model.guideline import GuidelineChunk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("verify_rag")

DEFAULT_QUERY = (
    "What dietary pattern and fiber intake is recommended to reduce LDL cholesterol?"
)

EXPECTED_TOPIC_KEYWORDS = (
    "diet",
    "dietary",
    "lipid",
    "cholesterol",
    "nutrition",
    "lifestyle",
    "prevention",
    "ldl",
    "cardiovascular",
)


class RetrievedChunk(NamedTuple):
    """Retrieved chunk data without embedding vectors."""

    id: str
    chunk_id: str
    guideline_title: str
    publication_year: int
    section_title: str
    content: str


class SearchResult(NamedTuple):
    """Tuple containing retrieved chunk and cosine similarity score."""

    chunk: RetrievedChunk
    similarity: float


def generate_query_embedding(
    query: str,
    client: OpenAI,
    model: str,
) -> list[float]:
    """Generate embedding vector for search query using ARC VT LLM."""
    logger.info("Generating embedding with model '%s' for query: '%s'", model, query)
    response = client.embeddings.create(input=[query], model=model)
    embedding = response.data[0].embedding
    assert len(embedding) == 2560, (
        f"Embedding dimension mismatch: expected 2560, got {len(embedding)}"
    )
    return embedding


async def execute_similarity_search(
    query_vector: list[float],
    limit: int = 5,
) -> tuple[list[SearchResult], float, float | None]:
    """Execute pgvector cosine similarity search and return results with latencies."""
    async with SessionLocal() as session:
        # Pre-warm connection so pooler / SSL negotiation is excluded from query latency
        await session.execute(select(1))

        # 1. Measure true database execution time on PostgreSQL via EXPLAIN (ANALYZE, FORMAT JSON)
        db_execution_time_ms: float | None = None
        try:
            vec_literal = "[" + ",".join(str(x) for x in query_vector) + "]"
            explain_stmt = text(
                "EXPLAIN (ANALYZE, FORMAT JSON) "
                "SELECT id FROM rag.guideline_chunks "
                "WHERE status = 'active' "
                f"ORDER BY embedding <=> '{vec_literal}'::halfvec "
                f"LIMIT {limit};"
            )
            explain_res = await session.execute(explain_stmt)
            plan_json = explain_res.scalar()
            if isinstance(plan_json, str):
                import json

                plan_json = json.loads(plan_json)
            if isinstance(plan_json, list) and plan_json:
                db_execution_time_ms = float(plan_json[0].get("Execution Time", 0.0))
        except (SQLAlchemyError, RuntimeError, ValueError) as exc:
            logger.debug("Could not run EXPLAIN (ANALYZE): %s", exc)

        # 2. Execute the actual query to fetch rows and measure client round-trip latency
        distance_expr = GuidelineChunk.embedding.cosine_distance(query_vector).label(
            "distance"
        )
        similarity_expr = (1 - distance_expr).label("similarity")

        stmt = (
            select(
                GuidelineChunk.id,
                GuidelineChunk.chunk_id,
                GuidelineChunk.guideline_title,
                GuidelineChunk.publication_year,
                GuidelineChunk.section_title,
                GuidelineChunk.content,
                similarity_expr,
            )
            .where(GuidelineChunk.status == "active")
            .order_by(distance_expr)
            .limit(limit)
        )

        start_time = time.perf_counter()
        result = await session.execute(stmt)
        rows = result.all()
        client_rtt_ms = (time.perf_counter() - start_time) * 1000.0

    search_results: list[SearchResult] = []
    for row in rows:
        chunk = RetrievedChunk(
            id=str(row[0]),
            chunk_id=str(row[1]),
            guideline_title=str(row[2]),
            publication_year=int(row[3]),
            section_title=str(row[4]),
            content=str(row[5]),
        )
        sim = float(row[6])
        search_results.append(SearchResult(chunk=chunk, similarity=sim))

    return search_results, client_rtt_ms, db_execution_time_ms


def validate_results(
    results: list[SearchResult],
    latency_ms: float,
    max_latency_ms: float = 25.0,
    is_db_latency: bool = False,
) -> None:
    """Validate search results against Step 9 acceptance criteria."""
    logger.info("--- Validating Acceptance Criteria ---")

    # Criterion 2: Minimum result count
    assert len(results) >= 3, (
        f"Failed: Retrieved fewer than 3 chunks (got {len(results)})."
    )
    logger.info("✓ Criterion 2 Passed: Retrieved %d chunks (>= 3).", len(results))

    # Criterion 3: Semantic similarity threshold
    top_similarity = results[0][1]
    assert top_similarity >= 0.70, (
        f"Failed: Top similarity {top_similarity:.4f} is below 0.70 threshold."
    )
    logger.info(
        "✓ Criterion 3 Passed: Top similarity is %.4f (>= 0.70).", top_similarity
    )

    # Criterion 4: Clinical domain correctness
    top_chunk = results[0][0]
    matched_keyword = any(
        kw in top_chunk.guideline_title.lower() or kw in top_chunk.section_title.lower()
        for kw in EXPECTED_TOPIC_KEYWORDS
    )
    assert matched_keyword, (
        f"Failed: Top result does not match clinical domain keywords.\n"
        f"Guideline: {top_chunk.guideline_title}\n"
        f"Section: {top_chunk.section_title}"
    )
    logger.info(
        "✓ Criterion 4 Passed: Authoritative clinical guideline matched ('%s' - '%s').",
        top_chunk.guideline_title,
        top_chunk.section_title,
    )

    # Criterion 5: Text integrity
    for i, res in enumerate(results):
        content = res.chunk.content
        assert content and content.strip(), f"Failed: Chunk {i} has empty content."
        assert len(content.strip()) >= 20, (
            f"Failed: Chunk {i} content too short ({len(content.strip())} chars)."
        )
        assert "\xef\xbf\xbd" not in content, (
            f"Failed: Chunk {i} contains replacement chars ()."
        )
        assert "\x00" not in content, f"Failed: Chunk {i} contains null bytes."
    logger.info(
        "✓ Criterion 5 Passed: All retrieved chunks have clean, valid Markdown text."
    )

    # Criterion 6: Latency budget
    latency_label = "Database execution time" if is_db_latency else "Query latency"
    logger.info(
        "%s: %.2f ms (Budget: %.2f ms)", latency_label, latency_ms, max_latency_ms
    )
    assert latency_ms <= max_latency_ms, (
        f"Failed: {latency_label} {latency_ms:.2f} ms exceeded budget of {max_latency_ms:.2f} ms."
    )
    logger.info(
        "✓ Criterion 6 Passed: %s %.2f ms is within budget.", latency_label, latency_ms
    )


def print_results(
    results: list[SearchResult],
    client_latency_ms: float,
    db_latency_ms: float | None = None,
) -> None:
    """Print retrieved chunks and excerpts to terminal."""
    print("\n" + "=" * 80)
    print("STEP 9: RAG RETRIEVAL VERIFICATION RESULTS")
    if db_latency_ms is not None:
        print(
            f"Database Query Execution Time (PostgreSQL)   : {db_latency_ms:.2f} ms (Budget: 25.00 ms)"
        )
    print(f"Client Round-Trip Network Latency (Internet) : {client_latency_ms:.2f} ms")
    print("=" * 80)

    for rank, res in enumerate(results, start=1):
        chunk, sim = res.chunk, res.similarity
        excerpt = chunk.content.replace("\n", " ")[:200]
        print(f"\n[Rank {rank}] Similarity: {sim:.4f}")
        print(f"  Guideline : {chunk.guideline_title} ({chunk.publication_year})")
        print(f"  Section   : {chunk.section_title}")
        print(f"  Chunk ID  : {chunk.chunk_id}")
        print(f"  Excerpt   : {excerpt}...")
    print("\n" + "=" * 80 + "\n")

    for rank, res in enumerate(results, start=1):
        chunk, sim = res.chunk, res.similarity
        excerpt = chunk.content.replace("\n", " ")[:200]
        print(f"\n[Rank {rank}] Similarity: {sim:.4f}")
        print(f"  Guideline : {chunk.guideline_title} ({chunk.publication_year})")
        print(f"  Section   : {chunk.section_title}")
        print(f"  Chunk ID  : {chunk.chunk_id}")
        print(f"  Excerpt   : {excerpt}...")
    print("\n" + "=" * 80 + "\n")


async def main() -> None:
    """Execute Step 9 verification."""
    parser = argparse.ArgumentParser(
        description="Verify RAG knowledge base retrieval against Step 9 criteria."
    )
    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help="Clinical search query to test",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--max-latency-ms",
        type=float,
        default=25.0,
        help="Maximum allowed query latency in milliseconds (default: 25.0)",
    )
    args = parser.parse_args()

    client = OpenAI(
        api_key=settings.arc_llm_key,
        base_url=settings.arc_llm_endpoint,
    )

    try:
        # Step 1: Generate embedding
        query_vector = generate_query_embedding(
            query=args.query,
            client=client,
            model=settings.embedding_model,
        )

        # Step 2: Query database
        results, client_rtt_ms, db_exec_ms = await execute_similarity_search(
            query_vector=query_vector,
            limit=args.top_k,
        )

        # Step 3: Print retrieved chunks
        print_results(
            results=results,
            client_latency_ms=client_rtt_ms,
            db_latency_ms=db_exec_ms,
        )

        # Step 4: Validate criteria
        # Use true database execution time if measured; otherwise client round-trip
        effective_latency_ms = db_exec_ms if db_exec_ms is not None else client_rtt_ms
        is_db = db_exec_ms is not None
        validate_results(
            results=results,
            latency_ms=effective_latency_ms,
            max_latency_ms=args.max_latency_ms,
            is_db_latency=is_db,
        )

        logger.info("All Step 9 acceptance criteria passed successfully.")
        sys.exit(0)

    except AssertionError as ae:
        logger.error("Step 9 Assertion Failure: %s", ae)
        sys.exit(1)
    except Exception:
        logger.exception("Step 9 Execution Error")
        sys.exit(1)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
