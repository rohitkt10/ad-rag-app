## Context
- This repository is a Python project.
- Authoritative docs live in `docs/`:
  - Product/spec: `docs/prd.md`
  - Execution plan/tasks: `docs/plan.md`

## Operating rules
- Make minimal, reviewable diffs. Do not refactor unless explicitly requested.
- Follow the closest-scoped instructions in this repo; if any instruction conflicts with `~/.gemini/GEMINI.md`, this repo-local file wins.
- Do not add new runtime dependencies without approval.
- Prefer deterministic behavior; no network calls in unit tests. Mock external I/O.
- Avoid expensive compute by default (large downloads, full training, long indexing runs) unless explicitly requested.
- Never claim to have run commands you did not run.

## Project commands (authoritative)
- Environment: use `.venv` for virtual environments.
- Use the repo’s `Makefile` commands when available:
  - `make test`
  - `make lint`
  - `make fmt`
  - `make typecheck`
- If `Makefile` is missing or incomplete, inspect `pyproject.toml` and propose the exact replacement commands.

## Code organization
- Keep pure logic in `src/<pkg>/core/` (no external I/O).
- Put external I/O behind interfaces in `src/<pkg>/adapters/` (LLMs, vector DB/FAISS, filesystem, HTTP, cloud).
- Keep the entrypoint thin (`src/<pkg>/main.py` or `src/<pkg>/api.py`).

## Output format
- Summarize: (1) what changed, (2) commands to verify, (3) manual checks (if any).
- When asked “what next,” propose exactly one next action with the exact command(s) to run.
- If something is ambiguous, propose 1–2 options and pick the simplest default.
