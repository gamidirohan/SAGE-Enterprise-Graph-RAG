# AGENTS.md

Purpose: keep agent work in this repo consistent, safe, and fast.

## 1) Project Ground Rules
- Use `uv` and the local `.venv` for all Python work.
- Run Python commands as `uv run python ...`.
- Keep runnable utilities in `scripts/`.
- Keep data inputs in `data/` and generated analysis in `results/`.
- Do not hardcode secrets or API keys in code.

## 2) Environment and Config
- Load env from repo-root `.env`.
- Neo4j config should support:
  - `NEO4J_URI`
  - `NEO4J_USERNAME` (fallback: `NEO4J_USER`)
  - `NEO4J_PASSWORD` (fallback: `NEO4J_Password`)
  - `NEO4J_DATABASE` (use when opening Neo4j sessions)
- Prefer path-safe code using `Path(__file__).resolve()` and repo-root-relative paths.

## 3) Coding Practices for This Repo
- Prefer minimal, targeted edits over broad rewrites.
- Match existing style and structure in touched files.
- Keep logging informative for long-running pipeline/evaluation scripts.
- Avoid silent failures; log meaningful errors.
- When adding dependencies:
  - update `pyproject.toml`
  - run `uv sync`
  - keep `uv.lock` in sync
- Keep docs/commands updated when moving files or changing invocation paths.

## 4) Validation Before Finishing
- At minimum, run syntax checks on modified Python files:
  - `uv run python -m py_compile <files...>`
- If behavior changes, run the smallest relevant command path to verify it.
- Write test cases for any new modules implemented, and verify by running them after implementation.
- Call out anything not verified.

## 5) Self-Adapting Rule (Important)
This file should evolve, but only when it materially reduces repeated mistakes.

Update `AGENTS.md` only if one of these is true:
- The same mistake happened 2+ times in this repo.
- A rule would prevent frequent path/env/script confusion.
- A recurring review comment keeps appearing.

When updating:
- Add one short rule, tied to a concrete recurring issue.
- Keep rules general and reusable, not tied to one temporary incident.
- Avoid adding noisy or one-off reminders.
- If a rule is stale or no longer useful, remove it.
