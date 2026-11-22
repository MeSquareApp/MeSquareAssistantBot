import uuid
from dataclasses import dataclass
from datetime import datetime

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class GuidelineChunk(Base):
    """Represents a chunk of a clinical guideline in rag.guideline_chunks."""

    __tablename__ = "guideline_chunks"
    __table_args__ = ({"schema": "rag"},)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    chunk_id: Mapped[str] = mapped_column(String, unique=True, index=True)
    guideline_title: Mapped[str] = mapped_column(String)
    publication_year: Mapped[int] = mapped_column(Integer)
    section_title: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(Text)
    embedding = mapped_column(HALFVEC(2560))
    status: Mapped[str] = mapped_column(String, default="active", index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# Backward-compatible alias
Guideline = GuidelineChunk


@dataclass(frozen=True)
class GuidelineMetadata:
    """Document metadata extracted from filename or header."""

    title: str
    year: int
    slug: str


@dataclass
class ChunkRecord:
    """In-memory chunk representation before and after embedding."""

    chunk_id: str
    guideline_title: str
    publication_year: int
    section_title: str
    content: str
    embedding: list[float] | None = None
    status: str = "active"
