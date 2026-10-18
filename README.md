# Veltrix PeopleOps Assistant

HR assistant API for the Veltrix Systems AI/ML Engineer assessment: answers policy questions from the HR corpus with citations and manages leave through permission-checked tools.

> Work in progress. Setup, architecture diagram, assumptions, known limitations and sample queries are completed at the end of the build (TASK.pdf §11.1).

## Quick start

1. Put the assessment data in `./data` (`corpus/`, `manifest.json`, `veltrix.db`).
2. `cp .env.example .env` and set `LLM_PROVIDER`, `LLM_MODEL` and the matching API key:

   | `LLM_PROVIDER` | API key variable | Example `LLM_MODEL` |
   |---|---|---|
   | `openai` | `OPENAI_API_KEY` | `gpt-4.1-mini` |
   | `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-4-5` |
   | `gemini` (or `google`) | `GOOGLE_API_KEY` | `gemini-2.5-flash` |
   | `nvidia` | `NVIDIA_API_KEY` (+ `LLM_BASE_URL`) | `openai/gpt-oss-20b` |
3. `docker compose up` → service on port 8000; `GET /health` returns 200 when ingestion is done.

**`./data` must be writable.** The service adds three tables (`sessions`, `messages`, `proposals`) to `veltrix.db` and switches it to SQLite WAL mode, which creates `veltrix.db-wal` and `veltrix.db-shm` next to it. The existing tables (`employees`, `leave_balances`, `leave_requests`) keep their schema; leave tools write rows to them as the spec requires. A missing `veltrix.db` is never created: `/health` stays 503 instead.

## Documentation

- [DESIGN.md](DESIGN.md) — design summary (required deliverable)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — full architecture
- [docs/DECISIONS_QA.md](docs/DECISIONS_QA.md) — design decisions and trade-offs
- [docs/TECH_STACK.md](docs/TECH_STACK.md) — stack and reasons
- [docs/PRODUCTION_PLAN.md](docs/PRODUCTION_PLAN.md) — production next steps (out of scope)
