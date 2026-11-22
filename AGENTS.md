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

# Python Clean Code Standards

## 1. Type Safety & Data Modeling
- Enforce strict type annotations on all function signatures and return types. Never use `Any`.
- Use `dataclasses` or `Pydantic` models instead of raw dictionaries for structured data.
- Use `Literal` types or `Enum` for fixed sets of string/integer values.
- Verify runtime types with `assert isinstance(...)` before using `typing.cast()`.

## 2. Functions & Architecture
- Keep functions small and single-purpose (under 30 lines). Avoid flag arguments (booleans that alter function behavior).
- Require keyword-only arguments (`*`) for functions taking 4 or more parameters.
- Apply the **Rule of Three**: do not create helper functions, base classes, or abstractions until logic is duplicated three times.
- Use absolute imports over relative imports.
- Defer heavy computation or I/O at import time; use lazy loading or dependency injection.

## 3. Error Handling & Control Flow
- Prefer **Look Before You Leap (LBYL)** over `try/except` blocks for predictable conditions (e.g., check `path.exists()` before resolving).
- Never swallow exceptions with bare `except:` or `except Exception: pass`. Always catch specific exceptions and chain tracebacks (`raise NewError(...) from err`).
- Avoid deeply nested `if/else` blocks; use early returns (guard clauses).

## 4. Naming, Formatting & Comments
- Use descriptive, domain-accurate `snake_case` names for variables/functions and `PascalCase` for classes. Replace magic numbers with module-level `UPPER_SNAKE_CASE` constants.
- Write Google-style docstrings for all public modules, classes, and functions explaining *why* and architectural context, not *what* the code obviously does.
- Keep line length to 88–120 characters.

## 5. Agent Guardrails & Verification
- **Strictly Prohibited:** Never add `# type: ignore`, `# noqa`, or linter suppression comments to bypass checks. Fix the root cause.
- Before finishing any task, run and ensure zero errors on:
  1. `ruff check --fix . && ruff format .`
  2. `mypy --strict .`
  3. `pytest`
