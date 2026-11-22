"""Ingestion script for clinical guideline PDFs.

Orchestrates the 4-step pipeline:
1. Extract (Docling) -> Markdown
2. Chunk (LangChain) -> ~600-token sections along heading boundaries
3. Embed (ARC VT Qwen3-Embedding-4B) -> 2560-dimensional embeddings
4. Load (SQLAlchemy) -> rag.guideline_chunks table in PostgreSQL
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from docling.document_converter import DocumentConverter
from openai import OpenAI

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.core.db import SessionLocal, engine
from app.utils.pdf_chunk_handler import (
    chunk_markdown,
    create_document_converter,
    embed_chunks,
    extract_markdown,
    load_chunks,
    parse_filename_metadata,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest_guidelines")


async def ingest_file(
    pdf_path: Path,
    converter: DocumentConverter,
    openai_client: OpenAI,
    dry_run: bool = False,
) -> int:
    """Run Extract -> Chunk -> Embed -> Load for a single PDF document."""
    metadata = parse_filename_metadata(pdf_path)
    logger.info("Processing: %s (Year: %d)", metadata.title, metadata.year)

    # 1. Extract
    markdown_text = extract_markdown(pdf_path, converter)

    # 2. Chunk
    chunks = chunk_markdown(markdown_text, metadata)
    if not chunks:
        logger.warning("No chunks generated for %s.", pdf_path.name)
        return 0

    # 3. Embed
    embed_chunks(chunks, openai_client, model=settings.embedding_model)

    # 4. Load
    if dry_run:
        logger.info("[Dry Run] Skipping DB write for %d chunks.", len(chunks))
        return len(chunks)

    async with SessionLocal() as session:
        return await load_chunks(chunks, session)


async def main() -> None:
    """CLI entrypoint for guideline ingestion."""
    parser = argparse.ArgumentParser(
        description="Extract, chunk, embed, and load clinical guideline PDFs into Supabase."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=PROJECT_ROOT / "data" / "guidelines",
        help="Path to a PDF file or directory of PDFs (default: data/guidelines)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract, chunk, and embed without inserting into database",
    )
    args = parser.parse_args()

    target_path = args.path.resolve()
    if target_path.is_file() and target_path.suffix.lower() == ".pdf":
        pdf_files = [target_path]
    elif target_path.is_dir():
        pdf_files = sorted(target_path.glob("*.pdf"))
    else:
        logger.error("Target path does not exist or has no PDFs: %s", target_path)
        sys.exit(1)

    if not pdf_files:
        logger.warning("No PDF files found in %s", target_path)
        return

    logger.info("Found %d PDF file(s) to process.", len(pdf_files))
    converter = create_document_converter(do_ocr=False)

    logger.info(
        "Using ARC VT LLM endpoint (%s) with embedding model '%s'",
        settings.arc_llm_endpoint,
        settings.embedding_model,
    )
    openai_client = OpenAI(
        api_key=settings.arc_llm_key,
        base_url=settings.arc_llm_endpoint,
    )

    total_chunks = 0
    failed_files: list[tuple[Path, str]] = []
    try:
        for pdf in pdf_files:
            try:
                total_chunks += await ingest_file(
                    pdf_path=pdf,
                    converter=converter,
                    openai_client=openai_client,
                    dry_run=args.dry_run,
                )
            except (RuntimeError, ValueError, OSError, TypeError) as e:
                logger.error("Failed to ingest %s: %s", pdf.name, e)
                failed_files.append((pdf, str(e)))

        logger.info(
            "Finished ingesting %d of %d PDFs (%d total chunks).",
            len(pdf_files) - len(failed_files),
            len(pdf_files),
            total_chunks,
        )
        if failed_files:
            logger.warning(
                "%d PDF(s) failed during ingestion:\n%s",
                len(failed_files),
                "\n".join(f" - {p.name}: {err}" for p, err in failed_files),
            )
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
