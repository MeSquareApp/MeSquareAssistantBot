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
