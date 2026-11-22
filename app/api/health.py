"""Health check API endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db

router = APIRouter(tags=["health"])

DbSession = Annotated[AsyncSession, Depends(get_db)]


@router.get(
    "/health",
    response_model=dict[str, str],
    status_code=status.HTTP_200_OK,
    summary="Service liveness probe",
)
async def health_check() -> dict[str, str]:
    """Execute service health probe confirming HTTP API availability."""
    return {"status": "healthy", "service": "MeSquareAssistantBot"}


@router.get(
    "/health/db",
    response_model=dict[str, str],
    status_code=status.HTTP_200_OK,
    summary="Database readiness probe",
)
async def db_health_check(db: DbSession) -> dict[str, str]:
    """Execute simple SELECT 1 query to confirm database connectivity."""
    await db.execute(text("SELECT 1"))
    return {"status": "healthy", "service": "MeSquareAssistantBot"}
