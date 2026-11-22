# FastAPI Project Rules

## Commands
- Dev server: `uv run fastapi dev`
- Prod server: `uv run fastapi run`
- Tests: `uv run pytest`

## Architecture & Routing
- Group endpoints using `APIRouter` (one router per module under `routers/`) and mount in `main.py` with `app.include_router(router, prefix="/api/v1")`.
- Always declare `response_model=` on route decorators for strict serialization and OpenAPI generation.
- Use explicit status constants (`status.HTTP_201_CREATED`, `status.HTTP_204_NO_CONTENT`) instead of hardcoded integers.
- Register global exception handlers via `app.add_exception_handler()` for consistent error payloads.

## Async vs. Sync (Strict)
- Use `async def` ONLY when all I/O inside the route is non-blocking and awaited (e.g., `httpx.AsyncClient`, async DB drivers).
- Use plain `def` for routes calling synchronous/blocking libraries (e.g., `requests`, `time.sleep`, sync ORMs) so FastAPI executes them in a threadpool without freezing the event loop.
- Never use `requests.get()` or `time.sleep()` inside an `async def` route.

## Dependency Injection & Pydantic v2
- Always use `typing.Annotated` for dependencies: `CurrentUser = Annotated[User, Depends(get_current_user)]`. Never use default-argument `Depends()` syntax.
- Use Pydantic v2 syntax only (`model_config = ConfigDict(...)`, `@field_serializer`). Do not use deprecated v1 `class Config:` or `json_encoders`.
- Offload non-blocking post-response tasks (emails, audit logs) to `BackgroundTasks`.

## Testing
- Use `fastapi.testclient.TestClient` with plain `def test_...` functions (or `httpx.AsyncClient` with `ASGITransport` for async tests).
- Mock dependencies using `app.dependency_overrides`.
