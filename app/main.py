"""Main FastAPI application initialization and routing."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.db_health import router as health_router
from app.core.db import engine


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifecycle and database connection cleanup."""
    yield
    await engine.dispose()


app = FastAPI(lifespan=lifespan)

# Register the routers to the application
app.include_router(health_router)


@app.get("/", response_model=dict[str, str])
async def root() -> dict[str, str]:
    """Root health check endpoint."""
    return {"message": "Hello World"}
