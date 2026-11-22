"""Database health check API endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db

router = APIRouter(tags=["health"])

DbSession = Annotated[AsyncSession, Depends(get_db)]


@router.get("/health")
async def health(db: DbSession) -> dict[str, str]:
    """Execute simple SELECT 1 query to confirm database connectivity."""
    await db.execute(text("SELECT 1"))
    return {"status": "healthy"}
