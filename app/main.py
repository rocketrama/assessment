"""FastAPI app: POST /chat and GET /health (TASK.pdf §9)."""

import asyncio
import logging
import os
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.schemas import ChatRequest, ChatResponse
from app.config import Settings, get_settings
from app.data.db import connect, init_app_tables
from app.data.employees import Employee, get_employee
from app.data.sessions import SessionOwnershipError, claim_session
from app.ingestion.pipeline import run_ingestion
from app.logging_setup import configure_logging

log = logging.getLogger("veltrix")


class AppState:
    def __init__(self) -> None:
        self.ready = False
        self.error: str | None = None


def create_app(
    settings: Settings | None = None,
    ingest: Callable[[Settings], None] = run_ingestion,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_format)
    state = AppState()

    def _startup() -> None:
        if not os.path.isfile(settings.db_path):
            # Never let sqlite3 create an empty DB in place of the reviewers' file.
            raise FileNotFoundError(f"database not found: {settings.db_path}")
        init_app_tables(settings.db_path)
        ingest(settings)

    async def _run_startup() -> None:
        try:
            await asyncio.to_thread(_startup)
            state.ready = True
            log.info("ingestion finished; service ready")
        except Exception as exc:
            state.error = f"{type(exc).__name__}: {exc}"
            log.exception("startup failed: %s", state.error)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Ingestion runs in the background so the port opens at once;
        # /health reports readiness (§9.2).
        task = asyncio.create_task(_run_startup())
        yield
        task.cancel()

    app = FastAPI(title="Veltrix PeopleOps Assistant", lifespan=lifespan)
    app.state.app_state = state

    @app.exception_handler(RequestValidationError)
    async def _invalid_body(request: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI defaults to 422; the contract says 400 for a missing or invalid body (§9.1).
        return JSONResponse(status_code=400, content={"detail": "missing or invalid request body"})

    @app.get("/health")
    def health() -> JSONResponse:
        if state.ready:
            return JSONResponse(status_code=200, content={"status": "ok"})
        return JSONResponse(status_code=503, content={"status": "starting"})

    def _requester(request: Request) -> Employee:
        # Header read as optional: a missing header must be 401, not FastAPI's 422.
        employee_id = (request.headers.get("X-Employee-Id") or "").strip()
        employee = None
        if employee_id:
            with connect(settings.db_path) as conn:
                employee = get_employee(conn, employee_id)
        if employee is None:
            raise HTTPException(status_code=401, detail="unknown or missing X-Employee-Id")
        return employee

    # Plain `def`: FastAPI runs it in the thread pool, so blocking SQLite work
    # never stalls the event loop.
    @app.post("/chat", response_model=ChatResponse)
    def chat(body: ChatRequest, request: Request) -> ChatResponse:
        if not state.ready:
            raise HTTPException(status_code=503, detail="service is starting")
        started = time.perf_counter()
        request_id = str(uuid.uuid4())
        requester = _requester(request)
        with connect(settings.db_path) as conn:
            try:
                claim_session(conn, body.session_id, requester.employee_id)
            except SessionOwnershipError:
                log.warning(
                    "session ownership mismatch",
                    extra={"request_id": request_id, "employee_id": requester.employee_id},
                )
                raise HTTPException(status_code=403, detail="session belongs to another employee")

        # Orchestration arrives in step 6; until then a contract-valid placeholder.
        response = ChatResponse(
            request_id=request_id,
            answer="The assistant is not fully available yet.",
            route="no_answer",
        )
        # IDs and counts only; the message and answer text are never logged.
        log.info(
            "chat handled",
            extra={
                "request_id": request_id,
                "employee_id": requester.employee_id,
                "route": response.route,
                "tool_calls": [t.name for t in response.tool_calls],
                "corpus_tokens": 0,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return response

    return app


app = create_app()
