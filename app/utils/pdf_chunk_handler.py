"""Helper functions for clinical guideline PDF extraction, chunking, embedding, and loading."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from openai import OpenAI
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.model.guideline import ChunkRecord, GuidelineChunk, GuidelineMetadata

logger = logging.getLogger(__name__)

HEADERS_TO_SPLIT_ON: list[tuple[str, str]] = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]


def parse_filename_metadata(pdf_path: Path) -> GuidelineMetadata:
    """Extract guideline title, publication year, and slug from filename."""
    stem = pdf_path.stem
    year_match = re.match(r"^(\d{4})[-_]", stem)
    year = int(year_match.group(1)) if year_match else 2025

    cleaned_title = re.sub(r"^\d{4}[-_]", "", stem)
    cleaned_title = cleaned_title.replace("-", " ").replace("_", " ").strip()
    words = cleaned_title.split()
    capitalized_title = " ".join(
        word.upper() if len(word) <= 4 and word.isalpha() else word.capitalize()
        for word in words
    )
    full_title = f"{year} {capitalized_title}".strip()

    slug = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
    return GuidelineMetadata(title=full_title, year=year, slug=slug)


def create_document_converter(do_ocr: bool = False) -> DocumentConverter:
    """Create a Docling converter optimized for digital PDFs.

    Disabling OCR for native digital PDFs accelerates processing by 5x to 10x.
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = do_ocr
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def extract_markdown(pdf_path: Path, converter: DocumentConverter) -> str:
    """Extract structured Markdown and tables from a multi-column PDF using Docling."""
    logger.info("Extracting Markdown from %s with Docling...", pdf_path.name)
    result = converter.convert(pdf_path)
    return result.document.export_to_markdown()


def chunk_markdown(
    markdown_text: str,
    metadata: GuidelineMetadata,
    chunk_size: int = 600,
    chunk_overlap: int = 60,
) -> list[ChunkRecord]:
    """Slice Markdown along heading boundaries into token-bounded sections."""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    header_sections = header_splitter.split_text(markdown_text)

    token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    documents = token_splitter.split_documents(header_sections)
    chunks: list[ChunkRecord] = []

    for index, doc in enumerate(documents, start=1):
        content = doc.page_content.strip()
        if not content:
            continue

        header_keys = ["Header 1", "Header 2", "Header 3", "Header 4"]
        found_headers = [
            doc.metadata[key] for key in header_keys if key in doc.metadata
        ]
        section_title = " > ".join(found_headers) if found_headers else "General"

        chunk_id = f"{metadata.slug}_{metadata.year}_{index:04d}"
        chunks.append(
            ChunkRecord(
                chunk_id=chunk_id,
                guideline_title=metadata.title,
                publication_year=metadata.year,
                section_title=section_title,
                content=content,
            )
        )

    logger.info("Created %d chunks from %s", len(chunks), metadata.title)
    return chunks


def embed_chunks(
    chunks: Sequence[ChunkRecord],
    client: OpenAI,
    model: str = "Qwen3-Embedding-4B",
    batch_size: int = 64,
) -> None:
    """Generate 2560-dimensional embeddings using ARC VT's Qwen3-Embedding-4B."""
    logger.info(
        "Generating embeddings using model=%s in batches of %d...", model, batch_size
    )
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [chunk.content for chunk in batch]
        try:
            response = client.embeddings.create(input=texts, model=model)
        except Exception as exc:
            logger.error(
                "Embedding call failed on endpoint %s with model '%s': %s",
                client.base_url,
                model,
                exc,
            )
            raise
        for chunk, item in zip(batch, response.data, strict=True):
            chunk.embedding = item.embedding
    logger.info("Successfully generated embeddings for %d chunks.", len(chunks))


async def load_chunks(
    chunks: Sequence[ChunkRecord],
    session: AsyncSession,
) -> int:
    """Idempotently insert or update guideline chunks in PostgreSQL."""
    if not chunks:
        return 0

    values = [
        {
            "chunk_id": chunk.chunk_id,
            "guideline_title": chunk.guideline_title,
            "publication_year": chunk.publication_year,
            "section_title": chunk.section_title,
            "content": chunk.content,
            "embedding": chunk.embedding,
            "status": chunk.status,
        }
        for chunk in chunks
    ]

    stmt = insert(GuidelineChunk).values(values)
    upsert_stmt = stmt.on_conflict_do_update(
        index_elements=["chunk_id"],
        set_={
            "guideline_title": stmt.excluded.guideline_title,
            "publication_year": stmt.excluded.publication_year,
            "section_title": stmt.excluded.section_title,
            "content": stmt.excluded.content,
            "embedding": stmt.excluded.embedding,
            "status": stmt.excluded.status,
        },
    )

    await session.execute(upsert_stmt)
    await session.commit()
    logger.info("Successfully loaded %d chunks into rag.guideline_chunks.", len(chunks))
    return len(chunks)
