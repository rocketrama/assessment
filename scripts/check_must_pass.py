"""Check the five must-pass rules (TASK.pdf §12.1) against the real model and corpus.

Usage:
    python scripts/check_must_pass.py            # full run, including a cold start (~3–5 min)
    python scripts/check_must_pass.py --warm     # reuse a copy of ./cache for check 1 (faster)

Everything runs on copies in a temporary folder: ./data and ./cache are never
changed. Needs the LLM settings in .env. Prints a PASS/FAIL table and exits 1
if any check fails.

  1. /health returns 200 within 10 minutes (timed from a cold cache).
  2. No document the requester may not read is ever sent to the LLM: every
     prompt of every chat request is recorded and searched for the text of
     every chunk the requester's role can't read.
  3. No employee sees leave data they may not see: every non-HR employee asks
     about every colleague; no tool call may target someone outside their
     visible set, and a reply about someone not visible must be the fixed
     refusal — or a reply that shows no data at all (e.g. "please try again"
     when the LLM failed), which is counted and reported separately.
  4. No LLM call carries more than 3,000 corpus tokens (from the call logs).
  5. Modified corpus: a value in a copy of a document is changed, the service
     is restarted on it, and the answer must change with it.
"""

import argparse
import json
import logging
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.data.employees import get_employee  # noqa: E402
from app.llm.client import CORPUS_TOKEN_LIMIT  # noqa: E402
from app.main import create_app  # noqa: E402
from app.retrieval.eligibility import can_read  # noqa: E402
from app.tools.authz import ACCESS_DENIED_MESSAGE, can_view_employee  # noqa: E402

HEALTH_LIMIT_SECONDS = 600
QUESTIONS = [
    "How many sick days do I get?",
    "What does the manager handbook say about approving leave?",
    "What are the annual salary bands for level L3?",
    "What are the required steps of a performance improvement plan?",
]
# Check 5: change one number in a copy of the India addendum's carry-forward rule.
MODIFIED_DOC = "leave_addendum_india.docx"
MODIFIED_FROM, MODIFIED_TO = "15 days", "20 days"
MODIFIED_QUESTION = ("E1001", "How many days of annual leave can I carry forward?")


class Recorder:
    """Wraps the chat provider: every prompt sent to the LLM is kept."""

    def __init__(self, inner):
        self.inner, self.prompts = inner, []

    def chat(self, model, system, user, max_tokens, timeout):
        self.prompts.append(system + "\n" + user)
        return self.inner.chat(model, system, user, max_tokens, timeout)


class CallLog(logging.Handler):
    def __init__(self):
        super().__init__()
        self.calls: list[dict] = []

    def emit(self, record):
        if record.getMessage() == "llm call":
            self.calls.append({"purpose": getattr(record, "purpose", ""), "corpus_tokens": getattr(record, "corpus_tokens", 0)})


def copy_inputs(work: Path, settings: Settings) -> Settings:
    data = work / "data"
    shutil.copytree(settings.corpus_dir, data / "corpus")
    shutil.copy(settings.manifest_path, data / "manifest.json")
    src, dst = sqlite3.connect(settings.db_path), sqlite3.connect(data / "veltrix.db")
    src.backup(dst)
    src.close()
    dst.close()
    return settings.model_copy(update={
        "corpus_dir": str(data / "corpus"), "manifest_path": str(data / "manifest.json"),
        "db_path": str(data / "veltrix.db"), "log_level": "WARNING",
    })


def start(settings: Settings):
    client = TestClient(create_app(settings))
    client.__enter__()
    started = time.monotonic()
    while client.get("/health").status_code != 200:
        if time.monotonic() - started > HEALTH_LIMIT_SECONDS:
            break
        time.sleep(1)
    return client, time.monotonic() - started


def chat(client, employee, message, session):
    resp = client.post("/chat", json={"session_id": session, "message": message}, headers={"X-Employee-Id": employee})
    return resp.status_code, resp.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warm", action="store_true", help="start from a copy of ./cache instead of a cold cache")
    args = parser.parse_args()

    base = Settings()
    results: list[tuple[str, bool, str]] = []
    work = Path(tempfile.mkdtemp(prefix="must-pass-"))
    try:
        settings = copy_inputs(work, base).model_copy(update={"cache_dir": str(work / "cache")})
        if args.warm and Path(base.cache_dir).exists():
            shutil.copytree(base.cache_dir, work / "cache")

        log = CallLog()
        logging.getLogger("veltrix.llm").addHandler(log)

        # 1. health
        client, seconds = start(settings)
        # The chat runs at WARNING; the LLM call records (INFO) are captured here only.
        llm_logger = logging.getLogger("veltrix.llm")
        llm_logger.setLevel(logging.INFO)
        llm_logger.propagate = False
        ok = client.get("/health").status_code == 200 and seconds <= HEALTH_LIMIT_SECONDS
        results.append(("1. /health 200 within 10 min", ok, f"{seconds:.0f} s ({'warm' if args.warm else 'cold'} cache)"))
        state = client.app.state.app_state
        corpus = state.retriever.corpus
        recorder = Recorder(state.llm.provider)
        state.llm.provider = recorder
        employees = ["E1001", "E1002", "E1003", "E1004", "M2001", "M2002", "H9001"]
        conn = sqlite3.connect(settings.db_path)
        conn.row_factory = sqlite3.Row
        people = {e: get_employee(conn, e) for e in employees}
        log.calls.clear()  # ingestion calls are not chat requests

        # 2. restricted documents never sent
        leaks, n = [], 0
        for emp in ("E1001", "M2001", "H9001"):
            role = people[emp].role
            hidden = [c for c in corpus.chunks.values() if not can_read(role, corpus.docs[c.doc_id].access_level)]
            for q in QUESTIONS:
                recorder.prompts.clear()
                chat(client, emp, q, f"mp2-{emp}-{n}")
                n += 1
                for chunk in hidden:
                    body = chunk.text.split("\n\n", 1)[-1][:120]
                    if any(body in p for p in recorder.prompts):
                        leaks.append(f"{emp}/{role}: {chunk.chunk_id}")
        results.append(("2. no restricted document sent to the LLM", not leaks,
                         f"{len(QUESTIONS)} questions × 3 roles, every prompt searched" if not leaks else "; ".join(leaks)))

        # 3. unauthorised leave data
        problems, asked, no_data = [], 0, 0
        for emp in employees:
            if people[emp].role == "hr":
                continue
            for other in employees:
                if other == emp:
                    continue
                visible = can_view_employee(conn, people[emp], other)
                status, body = chat(client, emp, f"What is {other}'s leave balance?", f"mp3-{emp}-{other}")
                asked += 1
                targets = {t["args"].get("employee_id") for t in body.get("tool_calls", [])} - {None}
                if any(not can_view_employee(conn, people[emp], t) for t in targets):
                    problems.append(f"{emp}→{other}: tool targeted {targets}")
                if not visible and body.get("answer") != ACCESS_DENIED_MESSAGE:
                    if body.get("route") in ("no_answer", "clarify") and not body.get("tool_calls"):
                        no_data += 1  # nothing shown, but not the refusal either: report it
                    else:
                        problems.append(f"{emp}→{other}: {body.get('route')} '{body.get('answer', '')[:60]}'")
        detail = f"{asked} cross-employee questions"
        if no_data:
            detail += f"; {no_data} got a no-data reply instead of the refusal (LLM unavailable)"
        results.append(("3. no unauthorised leave data", not problems, detail if not problems else "; ".join(problems[:5])))

        # 4. corpus token budget
        biggest = max((c["corpus_tokens"] for c in log.calls), default=0)
        results.append((f"4. ≤ {CORPUS_TOKEN_LIMIT} corpus tokens per LLM call", bool(log.calls) and biggest <= CORPUS_TOKEN_LIMIT,
                        f"{len(log.calls)} chat LLM calls logged, largest {biggest}"))
        client.__exit__(None, None, None)
        conn.close()

        # 5. modified corpus
        from docx import Document

        doc_path = Path(settings.corpus_dir) / MODIFIED_DOC
        document = Document(doc_path)
        changed = 0
        for paragraph in document.paragraphs:
            for run in paragraph.runs:
                if MODIFIED_FROM in run.text:
                    run.text = run.text.replace(MODIFIED_FROM, MODIFIED_TO)
                    changed += 1
        document.save(doc_path)
        client, _ = start(settings)
        emp, question = MODIFIED_QUESTION
        _, body = chat(client, emp, question, "mp5")
        client.__exit__(None, None, None)
        new_value = MODIFIED_TO.split()[0]
        ok = changed > 0 and new_value in body.get("answer", "") and MODIFIED_FROM.split()[0] not in body.get("answer", "")
        results.append(("5. modified corpus changes the answer", ok,
                        f"{MODIFIED_FROM} → {MODIFIED_TO} in {MODIFIED_DOC}; answer: {body.get('answer', '')[:90]!r}"))
    finally:
        llm_logger = logging.getLogger("veltrix.llm")
        llm_logger.handlers.clear()
        llm_logger.propagate = True
        shutil.rmtree(work, ignore_errors=True)

    print("\nMust-pass checks (TASK.pdf §12.1)")
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  — {detail}")
    sys.exit(0 if results and all(ok for _, ok, _ in results) else 1)


if __name__ == "__main__":
    main()
