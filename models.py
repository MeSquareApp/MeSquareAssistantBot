"""Domain models module for database entities and chunk structures.

Re-exports clinical guideline models defined in app.model.guideline
to maintain compatibility with workflow nodes and Phase 1 schemas.
"""

from app.model.guideline import (
    Base,
    ChunkRecord,
    Guideline,
    GuidelineChunk,
    GuidelineMetadata,
)

__all__ = [
    "Base",
    "ChunkRecord",
    "Guideline",
    "GuidelineChunk",
    "GuidelineMetadata",
]
