"""Root FastAPI entrypoint re-exporting the application from app.main."""

from app.main import app

__all__ = ["app"]
