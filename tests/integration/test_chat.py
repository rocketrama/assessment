"""POST /chat end to end with a scripted LLM (planner + composer).

Everything between the two LLM calls is real: ingestion, retrieval with
precedence, tools with permission checks, routes, citations and PII scrubbing.
"""

import json
import logging
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.ingestion.loaders.ocr import PageOcr
from app.ingestion.pipeline import run_ingestion
from app.llm.client import CORPUS_TOKEN_LIMIT, LLMClient
from app.logging_setup import JsonFormatter
from app.main import create_app
from app.orchestration import orchestrator as orch
from app.tools.authz import ACCESS_DENIED_MESSAGE
from tests.conftest import HashingEmbedder, wait_ready
from tests.fixtures.fixture_db import build_fixture_db
from tests.fixtures.policy_corpus import RegexFactProvider, write_corpus


def plan(intent="policy_question", **fields):
    return {"intent": intent, "policy_query": None, "about_date": None, "reads": [], "change": None,
            "many_employees": False, "missing": [], **fields}


class ChatScript:
    """Fake chat model. Planner replies come from `plans` (by message text);
    the composer cites every passage it was sent unless `compose` overrides."""

    def __init__(self):
        self.plans: dict[str, object] = {}
        self.compose = None
        self.composer_prompts: list[str] = []
        self.planner_prompts: list[str] = []
        self.delay = 0.0  # seconds per call; like a real SDK, the call fails at `timeout`
        self.timeouts: list[float] = []

    def chat(self, model, system, user, max_tokens, timeout):
        self.timeouts.append(timeout)
        if self.delay:
            time.sleep(min(self.delay, timeout))
            if self.delay > timeout:
                raise TimeoutError("provider took too long")
        if system.startswith("You plan"):
            self.planner_prompts.append(user)
            reply = self.plans[user.rsplit("Message:\n", 1)[1]]
            if isinstance(reply, Exception):
                raise reply
            return json.dumps(reply)
        self.composer_prompts.append(user)
        ids = [i for i in re.findall(r'<document id="([^"]+)">', user) if i != "rule-table"]
        if self.compose:
            return json.dumps(self.compose(ids, user))
        # Like a real answer: state the current / upcoming values from the rule table.
        values = re.findall(r'^\s*(?:current|CONFLICT between|upcoming from \S+): "([^"]+)"', user, re.MULTILINE)
        answer = "This applies to you: " + ", ".join(values) + "." if values else "Here is what applies to you."
        return json.dumps({"answer": answer, "used_chunk_ids": ids, "covered": "full"})


@pytest.fixture
def script() -> ChatScript:
    return ChatScript()


@contextmanager
def chat_app(tmp_path, script, extra_docs=None, retries=0, **overrides) -> Iterator[TestClient]:
    """The real app over the fixture DB and policy corpus, with a scripted chat LLM."""
    db = tmp_path / "veltrix.db"
    build_fixture_db(str(db))
    corpus, manifest = write_corpus(tmp_path, extra_docs)
    settings = Settings(_env_file=None, db_path=str(db), corpus_dir=str(corpus), manifest_path=str(manifest),
                        cache_dir=str(tmp_path / "cache"), app_today="2026-09-15", **overrides)

    def ingest(s):
        run_ingestion(s, PageOcr(vision=None, use_tesseract=False),
                      llm=LLMClient(RegexFactProvider(), "facts"), embedder=HashingEmbedder())

    with TestClient(create_app(settings, ingest=ingest, llm=LLMClient(script, "chat", retries=retries))) as c:
        wait_ready(c)
        yield c


@pytest.fixture
def client(tmp_path, script) -> Iterator[TestClient]:
    with chat_app(tmp_path, script) as c:
        yield c


def ask(client, script, message, plan_reply, employee="E1001", session="s1"):
    script.plans[message] = plan_reply
    resp = client.post("/chat", json={"session_id": session, "message": message}, headers={"X-Employee-Id": employee})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    check_route_invariants(body)
    return body


def check_route_invariants(body):
    """TASK.pdf §8.1 on every response."""
    route, cites, tools = body["route"], body["citations"], body["tool_calls"]
    expected = {
        "rag": (True, False), "tool": (False, True), "hybrid": (True, True),
        "clarify": (False, False), "no_answer": (False, False), "refuse": (False, False),
    }[route]
    assert (bool(cites), bool(tools)) == expected, body
    if body["pending_confirmation"]:
        assert any(t["status"] == "pending_confirmation" for t in tools)


def cited_docs(body):
    return sorted({c["doc_id"] for c in body["citations"]})


# --- documents --------------------------------------------------------------

def test_policy_question_cites_the_winning_policy_only(client, script):
    body = ask(client, script, "How many sick days do I get?", plan(policy_query="sick leave days per year"))
    assert body["route"] == "rag"
    # v2 wins; v3 is announced as upcoming; the outdated FAQ and v1 are never cited.
    assert cited_docs(body) == ["POL-V2", "POL-V3"]
    assert all(c["locator"] == "Sick leave" for c in body["citations"])
    table = script.composer_prompts[-1]
    assert '"12 days"' in table and "outranked" in table and "upcoming from 2027-01-01" in table


def test_past_date_question_uses_the_old_policy(client, script):
    body = ask(client, script, "What was sick leave in 2025?", plan(policy_query="sick leave days", about_date="in 2025"))
    assert cited_docs(body) == ["POL-V1"]
    assert '"question_is_about_date": "2025-12-31"' in script.composer_prompts[-1]


def test_restricted_document_never_sent_or_cited(client, script):
    message = "Confidential salary bands"
    for employee in ("E1001", "M2001"):
        body = ask(client, script, message, plan(policy_query=message), employee=employee, session=f"s-{employee}")
        assert "HR-BANDS" not in cited_docs(body)
        documents = script.composer_prompts[-1].split("<documents>", 1)[-1]
        assert "band L3 pays" not in documents  # the HR-only text never reaches the LLM


def test_composer_cannot_cite_what_it_was_not_sent(client, script):
    script.compose = lambda ids, user: {"answer": "x", "used_chunk_ids": ["HR-BANDS::000", "FAQ::000"], "covered": "full"}
    body = ask(client, script, "How many sick days do I get?", plan(policy_query="sick leave"))
    assert body["route"] == "no_answer" and body["answer"] == orch.NO_ANSWER


def test_nothing_covers_the_question_is_no_answer(client, script):
    script.compose = lambda ids, user: {"answer": "", "used_chunk_ids": [], "covered": "none"}
    body = ask(client, script, "What is the dress code?", plan(policy_query="dress code"))
    assert body["route"] == "no_answer" and "HR team" in body["answer"]


def test_corpus_tokens_logged_and_within_budget(client, script):
    handler = _Collect()
    logging.getLogger("veltrix").addHandler(handler)
    try:
        ask(client, script, "How many sick days do I get?", plan(policy_query="sick leave"))
    finally:
        logging.getLogger("veltrix").removeHandler(handler)
    composer = [e for e in handler.entries if e.get("purpose") == "composer"]
    chat = [e for e in handler.entries if e["msg"] == "chat handled"]
    assert composer and 0 < composer[-1]["corpus_tokens"] <= CORPUS_TOKEN_LIMIT
    assert chat[-1]["corpus_tokens"] == composer[-1]["corpus_tokens"]
    planner = [e for e in handler.entries if e.get("purpose") == "planner"]
    assert planner[-1]["corpus_tokens"] == 0  # the planner never sees documents


# --- tools ------------------------------------------------------------------

def test_own_balance_is_a_tool_answer(client, script):
    body = ask(client, script, "What's my balance?", plan("leave_info", reads=[{"tool": "get_leave_balance", "employee": "self"}]))
    assert body["route"] == "tool"
    assert body["tool_calls"] == [{"name": "get_leave_balance", "args": {"employee_id": "E1001"}, "status": "success"}]
    assert body["answer"] == "Your leave balances: annual 12 days, sick 6 days."
    assert script.composer_prompts == []  # records only: written in code


def test_future_balance_gets_the_calendar_in_code(client, script):
    ask(client, script, "How many annual days will I have by the end of December?",
        plan(policy_query="annual leave accrual", reads=[{"tool": "get_leave_balance", "employee": "self"}]))
    assert "4 (2026-09-30, 2026-10-31, 2026-11-30, 2026-12-31) month-end dates" in script.composer_prompts[-1]


def test_future_balance_looks_up_the_balance_when_the_planner_forgets(client, script):
    """Live finding: the planner sometimes returned only a carry-forward query, no balance read."""
    body = ask(client, script, "How many annual leave days will I have by the end of December?",
               plan(policy_query="annual leave carry forward"))
    assert body["tool_calls"][0] == {"name": "get_leave_balance", "args": {"employee_id": "E1001"}, "status": "success"}
    assert "Future-balance calendar" in script.composer_prompts[-1]


def test_no_calendar_without_a_future_date(client, script):
    ask(client, script, "Sick days left and the rule?",
        plan(policy_query="sick leave", reads=[{"tool": "get_leave_balance", "employee": "self", "leave_type": "sick"}]))
    assert "Future-balance calendar" not in script.composer_prompts[-1]


def test_policy_plus_tools_is_hybrid(client, script):
    body = ask(client, script, "Sick days left and the rule?",
               plan(policy_query="sick leave", reads=[{"tool": "get_leave_balance", "employee": "self", "leave_type": "sick"}]))
    assert body["route"] == "hybrid"
    assert body["tool_calls"][0]["args"] == {"employee_id": "E1001", "leave_type": "sick"}


@pytest.mark.parametrize("target", ["E1003", "E9999", "Test Employee De", "Nobody Atall"])
def test_others_data_gets_the_same_fixed_refusal(client, script, target):
    body = ask(client, script, f"Balance of {target}?", plan("leave_info", reads=[{"tool": "get_leave_balance", "employee": target}]))
    assert body["route"] == "refuse" and body["answer"] == ACCESS_DENIED_MESSAGE


def test_manager_can_ask_about_a_report_by_name(client, script):
    body = ask(client, script, "Balance of Test Employee Us?",
               plan("leave_info", reads=[{"tool": "get_leave_balance", "employee": "Test Employee Us"}]), employee="M2001")
    assert body["route"] == "tool" and body["tool_calls"][0]["args"] == {"employee_id": "E1002"}


def test_ambiguous_name_asks_which_one(client, script):
    body = ask(client, script, "Balance of Test Employee?",
               plan("leave_info", reads=[{"tool": "get_leave_balance", "employee": "Test Employee"}]), employee="H9001")
    assert body["route"] == "clarify" and "E1001" in body["answer"] and "E1004" in body["answer"]


def test_team_leave_next_week_for_a_manager(client, script):
    body = ask(client, script, "Who is off next week?",
               plan("leave_info", reads=[{"tool": "get_team_leave", "start": "next week"}]), employee="M2001")
    assert body["route"] == "tool"
    assert body["tool_calls"][0]["args"] == {"start_date": "2026-09-21", "end_date": "2026-09-25"}


def test_team_leave_next_month_is_the_whole_month(client, script):
    """Live finding: "next month" was answered with "Which dates should I check?"."""
    body = ask(client, script, "Is anyone in my team off next month?",
               plan("leave_info", reads=[{"tool": "get_team_leave", "start": "next month"}]), employee="M2001")
    assert body["route"] == "tool"
    assert body["tool_calls"][0]["args"] == {"start_date": "2026-10-01", "end_date": "2026-10-31"}


@pytest.mark.parametrize("planned", ["self", "Test Employee Us's"])
def test_a_name_in_the_message_beats_self(client, script, planned):
    """Live finding: "What is Test Employee Us's leave balance?" returned the manager's own balance."""
    body = ask(client, script, "What is Test Employee Us's leave balance?",
               plan("leave_info", reads=[{"tool": "get_leave_balance", "employee": planned}]), employee="M2001")
    assert body["route"] == "tool" and body["tool_calls"][0]["args"] == {"employee_id": "E1002"}


def test_a_name_the_requester_may_not_see_is_refused_not_swapped_for_self(client, script):
    body = ask(client, script, "What is Test Employee De's leave balance?",
               plan("leave_info", reads=[{"tool": "get_leave_balance", "employee": "self"}]))
    assert body["route"] == "refuse" and body["answer"] == ACCESS_DENIED_MESSAGE


def test_own_full_name_is_still_self(client, script):
    body = ask(client, script, "What is Test Employee In's balance?",
               plan("leave_info", reads=[{"tool": "get_leave_balance", "employee": "self"}]))
    assert body["tool_calls"][0]["args"] == {"employee_id": "E1001"}


def test_team_leave_refused_for_non_managers(client, script):
    body = ask(client, script, "Who is off next week?", plan("leave_info", reads=[{"tool": "get_team_leave", "start": "next week"}]))
    assert body["route"] == "refuse" and body["answer"] == orch.NOT_A_MANAGER


def test_ambiguous_date_asks(client, script):
    body = ask(client, script, "Team leave on 03/04?",
               plan("leave_info", reads=[{"tool": "get_team_leave", "start": "03/04"}]), employee="M2001")
    assert body["route"] == "clarify" and "03/04" in body["answer"]


# --- guardrails -------------------------------------------------------------

@pytest.mark.parametrize(
    "plan_reply, answer",
    [
        (plan("off_topic"), orch.OFF_TOPIC),
        (plan("manipulation"), orch.MANIPULATION),
        (plan("leave_info", many_employees=True, reads=[{"tool": "get_leave_balance", "employee": "E1002"}]), orch.BULK),
        (plan("leave_info", reads=[{"tool": "get_leave_balance", "employee": "E1002"},
                                   {"tool": "get_leave_balance", "employee": "E1003"}]), orch.BULK),
    ],
)
def test_refusals(client, script, plan_reply, answer):
    body = ask(client, script, "some message", plan_reply, employee="H9001")
    assert body["route"] == "refuse" and body["answer"] == answer


def test_missing_details_are_asked_in_one_message(client, script):
    body = ask(client, script, "Show leave", plan("leave_info", missing=["which employee", "which dates"]))
    assert body["route"] == "clarify" and "which employee, which dates" in body["answer"]


def test_planner_failure_is_a_safe_answer(client, script):
    body = ask(client, script, "Hello?", TimeoutError("slow"))
    assert body["route"] == "no_answer" and body["answer"] == orch.UNAVAILABLE


# --- personal data ----------------------------------------------------------

def test_answer_is_scrubbed(client, script):
    script.compose = lambda ids, user: {
        "answer": "12 days. Ask Test Employee De at e.de@veltrix.example or +00-000-0006. You are Test Employee In.",
        "used_chunk_ids": ids, "covered": "full",
    }
    body = ask(client, script, "Who do I ask?", plan(policy_query="sick leave"))
    assert "Test Employee De" not in body["answer"]  # not visible to E1001
    assert "Test Employee In" in body["answer"]  # their own name
    assert "e***@veltrix.example" in body["answer"] and "+00-000-0006" not in body["answer"]


def test_history_is_passed_to_the_planner(client, script):
    ask(client, script, "First question", plan("off_topic"))
    seen = {}

    def capture(model, system, user, max_tokens, timeout, _orig=script.chat):
        if system.startswith("You plan"):
            seen["user"] = user
        return _orig(model, system, user, max_tokens, timeout)

    script.chat = capture
    ask(client, script, "Second question", plan("off_topic"))
    assert "user: First question" in seen["user"] and orch.OFF_TOPIC[:20] in seen["user"]


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__()
        self.entries: list[dict] = []

    def emit(self, record):
        self.entries.append(json.loads(JsonFormatter().format(record)))


def test_team_dates_taken_from_the_message_when_the_planner_drops_them(client, script):
    body = ask(client, script, "Who in my team is on leave next week?",
               plan("leave_info", reads=[{"tool": "get_team_leave"}]), employee="M2001")
    assert body["route"] == "tool"
    assert body["tool_calls"][0]["args"] == {"start_date": "2026-09-21", "end_date": "2026-09-25"}


def test_id_in_the_message_beats_a_planner_self(client, script):
    """Live finding: the model planned 'self' for "What is E1003's leave balance?"."""
    body = ask(client, script, "What is E1003's leave balance?",
               plan("leave_info", reads=[{"tool": "get_leave_balance", "employee": "self"}]))
    assert body["route"] == "refuse" and body["answer"] == ACCESS_DENIED_MESSAGE


def test_own_id_in_the_message_is_still_self(client, script):
    body = ask(client, script, "What is E1001's leave balance?",
               plan("leave_info", reads=[{"tool": "get_leave_balance", "employee": "self"}]))
    assert body["tool_calls"][0]["args"] == {"employee_id": "E1001"}


def test_empty_composer_answer_is_never_sent(client, script):
    script.compose = lambda ids, user: {"answer": "  ", "used_chunk_ids": [], "covered": "full"}
    body = ask(client, script, "My balance and the rules?",
               plan(policy_query="annual leave", reads=[{"tool": "get_leave_balance", "employee": "self"}]))
    assert body["route"] == "no_answer" and body["answer"].strip()


def test_team_range_from_the_message_beats_the_plan(client, script):
    """Eval finding: for "between 5 and 9 October" the planner passed only "9 October"."""
    body = ask(client, script, "Who in my team is on leave between 5 and 9 October?",
               plan("leave_info", reads=[{"tool": "get_team_leave", "start": "9 October"}]), employee="M2001")
    assert body["tool_calls"][0]["args"] == {"start_date": "2026-10-05", "end_date": "2026-10-09"}


def test_spurious_missing_items_do_not_block_a_lookup(client, script):
    """Eval finding: the planner listed "leave type" as missing for a balance lookup."""
    body = ask(client, script, "What's my balance?",
               plan("leave_info", reads=[{"tool": "get_leave_balance", "employee": "self"}], missing=["leave type"]))
    assert body["route"] == "tool"


def test_superseded_value_in_the_answer_is_rewritten_once(client, script):
    """Layer 3: the answer names v1's / the FAQ's outdated 10 days; the composer is sent back
    once with exactly that value, and the rewrite is what the employee gets."""
    replies = iter([
        "You get 12 days of sick leave (POL-V2::000). Older sources said 10 days.",
        "You get 12 days of sick leave.",
    ])
    script.compose = lambda ids, user: {"answer": next(replies), "used_chunk_ids": ids, "covered": "full"}
    body = ask(client, script, "How many sick days do I get?", plan(policy_query="sick leave days per year"))
    assert len(script.composer_prompts) == 2
    assert "Your previous answer was rejected" in script.composer_prompts[1] and "(10 days)" in script.composer_prompts[1]
    assert body["answer"] == "You get 12 days of sick leave." and body["route"] == "rag"


def test_internal_ids_never_reach_the_employee_even_without_a_retry(client, script):
    script.compose = lambda ids, user: {"answer": "You get 12 days (POL‑V2::000).", "used_chunk_ids": ids, "covered": "full"}
    body = ask(client, script, "How many sick days do I get?", plan(policy_query="sick leave days per year"))
    assert body["answer"] == "You get 12 days." and len(script.composer_prompts) == 1
