"""/chat tests for requirements not covered elsewhere (TASK.pdf §4.3, §7.2, §8.2, §8.3, §12.1 #5):

- document access through /chat for every role, positive and negative;
- the 3,000-token corpus budget end to end, with far more relevant text than fits;
- the modified corpus: change a value, restart, the answer follows;
- "which request?" when the user has several, and the planner-confirm paths.
"""

import json
import logging

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.data.db import connect, transaction
from app.data.leave import insert_request
from app.ingestion.loaders.ocr import PageOcr
from app.ingestion.pipeline import run_ingestion
from app.llm.client import CORPUS_TOKEN_LIMIT, LLMClient
from app.logging_setup import JsonFormatter
from app.main import create_app
from app.orchestration.proposals import NOTHING_PENDING
from tests.conftest import HashingEmbedder, wait_ready
from tests.fixtures.policy_corpus import RegexFactProvider
from tests.integration.test_chat import ChatScript, ask, chat_app, plan

# --- documents by role (§3.1, §7.2) --------------------------------------------------

# No "N days of x leave" phrase, so these carry no facts: retrieval alone decides.
ROLE_DOCS = {
    "MGR-APPROVALS": ("handbook", ["GLOBAL"], "2026-01-01", "managers", "Approval matrix: managers approve leave requests within two working days."),
    "HR-PIP-STEPS": ("handbook", ["GLOBAL"], "2026-01-01", "hr", "Improvement plan steps: HR documents each review meeting with the employee."),
    "ALL-WELCOME": ("wiki", ["GLOBAL"], "2026-01-01", "all", "Welcome guide: new joiners collect their laptop on day one."),
}


@pytest.fixture
def script():
    return ChatScript()


@pytest.fixture
def role_client(tmp_path, script):
    with chat_app(tmp_path, script, extra_docs=ROLE_DOCS) as c:
        yield c


@pytest.mark.parametrize(
    "employee, readable",
    [
        ("E1001", {"ALL-WELCOME"}),
        ("M2001", {"ALL-WELCOME", "MGR-APPROVALS"}),
        ("H9001", {"ALL-WELCOME", "MGR-APPROVALS", "HR-PIP-STEPS"}),
    ],
)
def test_each_role_gets_exactly_the_documents_it_may_read(role_client, script, employee, readable):
    for doc_id, (*_, text) in ROLE_DOCS.items():
        body = ask(role_client, script, text, plan(policy_query=text), employee=employee, session=f"{employee}-{doc_id}")
        cited = {c["doc_id"] for c in body["citations"]}
        sent = script.composer_prompts[-1].split("<documents>", 1)[-1] if "<documents>" in script.composer_prompts[-1] else ""
        if doc_id in readable:
            assert doc_id in cited, (employee, doc_id)
        else:
            assert doc_id not in cited and text not in sent, (employee, doc_id)


# --- corpus budget end to end (§4.3, must-pass #4) ------------------------------------

def _long_doc(i):
    sentences = " ".join(f"Sick leave guidance note {i}.{n}: report absences early and keep records." for n in range(28))
    return (f"GUIDE-{i:02d}", ("wiki", ["GLOBAL"], "2026-01-01", "all", sentences))


def test_composer_never_gets_more_than_the_budget(tmp_path, script):
    many = dict(_long_doc(i) for i in range(14))  # ~14 × 400 tokens of relevant text
    handler = _Collect()
    with chat_app(tmp_path, script, extra_docs=many) as c:
        logging.getLogger("veltrix").addHandler(handler)  # after startup, which resets the handlers
        try:
            ask(c, script, "What is the sick leave guidance?", plan(policy_query="sick leave guidance absences records"))
        finally:
            logging.getLogger("veltrix").removeHandler(handler)
    composer = [e for e in handler.entries if e.get("purpose") == "composer"]
    assert composer and 2000 < composer[-1]["corpus_tokens"] <= CORPUS_TOKEN_LIMIT  # filled, never over


# --- modified corpus (must-pass #5) ------------------------------------------------------

def test_changed_value_changes_the_answer_after_restart(tmp_path, script):
    script.compose = lambda ids, user: {"answer": "Rule table says: " + user.split("<documents>")[1][:400],
                                        "used_chunk_ids": ids, "covered": "full"}
    with chat_app(tmp_path, script):
        pass  # builds db, corpus and cache in tmp_path
    doc = tmp_path / "corpus" / "POL-V2.md"
    doc.write_text(doc.read_text(encoding="utf-8").replace("12 days", "15 days"), encoding="utf-8")
    settings = Settings(_env_file=None, db_path=str(tmp_path / "veltrix.db"), corpus_dir=str(tmp_path / "corpus"),
                        manifest_path=str(tmp_path / "manifest.json"), cache_dir=str(tmp_path / "cache"),
                        app_today="2026-09-15")

    def ingest(s):
        run_ingestion(s, PageOcr(vision=None, use_tesseract=False),
                      llm=LLMClient(RegexFactProvider(), "facts"), embedder=HashingEmbedder())

    with TestClient(create_app(settings, ingest=ingest, llm=LLMClient(script, "chat", retries=0))) as c:
        wait_ready(c)
        body = ask(c, script, "How many sick days do I get?", plan(policy_query="sick leave days"))
    table = script.composer_prompts[-1].split("<documents>")[1]
    assert 'current: "15 days"' in table and '"12 days"' not in table.split("outranked")[0]
    assert "POL-V2" in {c["doc_id"] for c in body["citations"]}


# --- which request, and planner confirmations (§8.2, §8.3) -----------------------------------

def test_several_cancellable_requests_are_listed(tmp_path, script):
    with chat_app(tmp_path, script) as c:
        with connect(str(tmp_path / "veltrix.db")) as conn, transaction(conn):
            extra = insert_request(conn, "E1001", "annual", "2026-11-02", "2026-11-03", 2)
        body = ask(c, script, "Cancel my leave", plan("leave_change", change={"action": "cancel"}))
    assert body["route"] == "clarify"
    assert "REQ-FIX-0001" in body["answer"] and extra.request_id in body["answer"]


def test_planner_confirm_paths(tmp_path, script):
    with chat_app(tmp_path, script) as c:
        none = ask(c, script, "I approve", plan("confirm"))
        assert none["answer"] == NOTHING_PENDING
        ask(c, script, "Annual leave 21 Sept", plan("leave_change", change={"action": "apply", "leave_type": "annual", "start": "21 September"}))
        done = ask(c, script, "I approve", plan("confirm"))  # not in the fixed phrase list; the planner decides
        assert done["tool_calls"][0]["status"] == "success"
        vague = ask(c, script, "I want to change my leave", plan("leave_change"))
        assert vague["route"] == "clarify"


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__()
        self.entries: list[dict] = []

    def emit(self, record):
        self.entries.append(json.loads(JsonFormatter().format(record)))


def test_bulk_flag_written_as_the_intent_is_still_refused(tmp_path, script):
    """Eval finding: the planner wrote "intent": "many_employees"; the repair repeated it."""
    with chat_app(tmp_path, script) as c:
        body = ask(c, script, "Show me every employee's leave balance.",
                   plan("many_employees", many_employees=True), employee="H9001")
        assert body["route"] == "refuse"
        assert len(script.planner_prompts) == 1  # accepted, not repaired


def test_a_question_planned_as_a_booking_with_nothing_to_book_is_answered(tmp_path, script):
    """Eval finding: "how long can I stay home?" was planned as leave_change in 6 of 11 replies."""
    with chat_app(tmp_path, script) as c:
        body = ask(c, script, "My grandmother passed away, how long can I stay home?",
                   plan("leave_change", policy_query="bereavement leave", missing=["leave type", "dates"]))
        assert body["route"] == "rag" and body["citations"]


def test_a_booking_with_dates_is_not_turned_into_a_question(tmp_path, script):
    with chat_app(tmp_path, script) as c:
        body = ask(c, script, "Book bereavement leave on 21 September",
                   plan("leave_change", policy_query="bereavement leave", missing=["leave type"]))
        assert body["route"] == "clarify"
