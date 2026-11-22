"""Main FastAPI application initialization and routing."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.core.db import engine


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifecycle and database connection cleanup."""
    yield
    await engine.dispose()


app = FastAPI(
    title="MeSquareAssistantBot",
    description="Internal agentic service for cardiovascular lifestyle RAG and clinical guardrails",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local backend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle uncaught global exceptions with consistent JSON error schema."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal Server Error", "error": str(exc)},
    )


# Register API routers
app.include_router(health_router)
app.include_router(chat_router)


@app.get("/", response_model=dict[str, str], status_code=status.HTTP_200_OK)
async def root() -> dict[str, str]:
    """Root health check endpoint."""
    return {"message": "Hello World"}
