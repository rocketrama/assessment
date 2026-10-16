# Veltrix PeopleOps Assistant

HR assistant API for the Veltrix Systems AI/ML Engineer assessment: answers policy questions from the HR corpus with citations and manages leave through permission-checked tools.

> Work in progress. Setup, architecture diagram, assumptions, known limitations and sample queries are completed at the end of the build (TASK.pdf §11.1).

## Quick start

1. Put the assessment data in `./data` (`corpus/`, `manifest.json`, `veltrix.db`).
2. `cp .env.example .env` and set `LLM_PROVIDER`, `LLM_MODEL` and the matching API key.
3. `docker compose up` → service on port 8000; `GET /health` returns 200 when ingestion is done.

## Documentation

- [DESIGN.md](DESIGN.md) — design summary (required deliverable)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — full architecture
- [docs/DECISIONS_QA.md](docs/DECISIONS_QA.md) — design decisions and trade-offs
- [docs/TECH_STACK.md](docs/TECH_STACK.md) — stack and reasons
- [docs/PRODUCTION_PLAN.md](docs/PRODUCTION_PLAN.md) — production next steps (out of scope)
