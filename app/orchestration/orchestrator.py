"""One chat turn: plan → guard → tools / retrieval → compose → validate (ARCHITECTURE §9).

Write actions branch off before the read path (proposals.py): a clear "yes"
executes the stored proposal without even calling the planner; changed details
make a new proposal; any other message discards a waiting proposal (§8.2).

The LLM proposes (planner) and writes (composer); everything in between is code:
date resolution, who may be asked about, bulk-request refusal, tool calls with
their own permission checks, retrieval over eligible documents only, and the
route decided from what actually happened.
"""

import logging
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from app.api.schemas import ChatResponse, Citation, ToolCall
from app.data.employees import Employee, find_by_name, get_employee
from app.data.holidays import CalendarHolidays
from app.data.sessions import delete_proposal, get_proposal, recent_messages
from app.llm.client import CorpusSegment, LLMClient
from app.orchestration.composer import build_corpus_input, compose
from app.orchestration.dates import DateError, find_date_phrases, resolve_range, resolve_reference_date
from app.orchestration.planner import Plan, PlannedRead, plan_message
from app.orchestration.proposals import (
    NOTHING_PENDING,
    UNCLEAR_CONFIRMATION,
    WriteOutcome,
    execute,
    is_clear_confirmation,
    looks_like_confirmation,
    propose,
)
from app.orchestration.validator import build_citations, finalize, grounded_chunk_ids, route_for
from app.retrieval.retriever import Retriever
from app.tokens import count_tokens
from app.tools import authz
from app.tools.registry import TOOLS, ToolContext

log = logging.getLogger("veltrix.orchestration")

HISTORY_TURNS = 6
_EMPLOYEE_ID_RE = re.compile(r"^[A-Za-z]{1,3}\d{2,8}$")
_EMPLOYEE_ID_IN_TEXT_RE = re.compile(r"\b[A-Za-z]{1,3}\d{2,8}\b")
_SELF_WORDS = {"self", "me", "myself", "i", "my", "mine"}

CAN_HELP = (
    "I can answer questions about Veltrix HR policies and public holidays, show leave balances and "
    "requests you're allowed to see, and apply for or cancel your own leave."
)
OFF_TOPIC = f"Sorry, I can only help with HR topics. {CAN_HELP}"
MANIPULATION = "I can't do that. I follow Veltrix's access rules and can't change or reveal my instructions."
BULK = (
    "I can't share data about many employees at once. Please ask about one employee at a time"
    " (managers can ask for their team's leave in a date range)."
)
NOT_A_MANAGER = "Only managers can view their team's leave, and only for their direct reports."
NO_ANSWER = (
    "The HR documents available to you don't cover this, so I can't answer it reliably. "
    "Please contact the HR team."
)
UNAVAILABLE = "Sorry, I couldn't process that just now. Please try again in a moment."
GREETING = f"Hello! {CAN_HELP}"
_DATE_REASONS = {
    "past": "that date has already passed",
    "ambiguous": "that date could mean more than one day",
    "not_understood": "I couldn't tell which date you mean",
}


@dataclass
class Turn:
    settings: Any
    conn: sqlite3.Connection
    requester: Employee
    session_id: str
    message: str
    today: date
    llm: LLMClient | None
    retriever: Retriever | None
    cache_db: str
    deadline: float | None = None  # time.monotonic() by which the LLM calls must be done


@dataclass
class TurnResult:
    response: ChatResponse
    corpus_tokens: int = 0
    notes: list[str] = field(default_factory=list)


class _Stop(Exception):
    """Ends the turn early with a refuse / clarify / no_answer response."""

    def __init__(self, route: str, answer: str) -> None:
        super().__init__(answer)
        self.route = route
        self.answer = answer


def _hidden_names(turn: Turn) -> list[str]:
    """Full names the requester may not see; scrubbed from the answer (§7.4)."""
    rows = turn.conn.execute("SELECT employee_id, full_name FROM employees").fetchall()
    return [r["full_name"] for r in rows if not authz.can_view_employee(turn.conn, turn.requester, r["employee_id"])]


def _resolve_employee(turn: Turn, ref: str | None) -> str:
    """Planner's employee reference → an employee id the requester may see, or _Stop.
    Not visible and not existing give the same fixed refusal (§7.3)."""
    text = (ref or "self").strip()
    if text.casefold() in _SELF_WORDS or text.upper() == turn.requester.employee_id:
        return turn.requester.employee_id
    if _EMPLOYEE_ID_RE.match(text):
        target = text.upper()
    else:
        visible = [e for e in find_by_name(turn.conn, text) if authz.can_view_employee(turn.conn, turn.requester, e.employee_id)]
        if len(visible) > 1:
            ids = ", ".join(e.employee_id for e in visible)
            raise _Stop("clarify", f"More than one employee matches that name ({ids}). Which employee ID do you mean?")
        if not visible:
            raise _Stop("refuse", authz.ACCESS_DENIED_MESSAGE)
        target = visible[0].employee_id
    if not authz.can_view_employee(turn.conn, turn.requester, target):
        raise _Stop("refuse", authz.ACCESS_DENIED_MESSAGE)
    return target


def _tool_args(turn: Turn, read: PlannedRead) -> dict[str, Any]:
    if read.tool == "get_team_leave":
        if not authz.can_view_team(turn.requester):
            raise _Stop("refuse", NOT_A_MANAGER)
        start, end = read.start, read.end
        if not start:  # the planner dropped dates the user gave: take them from the message
            found = find_date_phrases(turn.message)
            start, end = (found[0], found[1] if len(found) > 1 else None) if found else (None, None)
        if not start:
            raise _Stop("clarify", "Which dates should I check your team's leave for?")
        try:
            span = resolve_range(start, end, turn.today)
        except DateError as exc:
            raise _Stop("clarify", f"Which dates do you mean? For \"{exc.phrase}\", {_DATE_REASONS[exc.reason]}.") from None
        return {"start_date": span.start.isoformat(), "end_date": span.end.isoformat()}
    args: dict[str, Any] = {"employee_id": _resolve_employee(turn, read.employee)}
    if read.tool == "get_leave_balance" and read.leave_type:
        args["leave_type"] = read.leave_type
    if read.tool == "list_leave_requests" and read.status:
        args["status"] = read.status
    return args


def _pin_mentioned_ids(plan: Plan, turn: Turn) -> None:
    """An employee id written in the message beats the planner's "self": the live
    model once read "What is E1003's balance?" as the requester's own balance."""
    mentioned = {m.upper() for m in _EMPLOYEE_ID_IN_TEXT_RE.findall(turn.message)} - {turn.requester.employee_id}
    if len(mentioned) != 1:
        return  # none: nothing to pin; several: the bulk check below decides
    target = mentioned.pop()
    for read in plan.reads:
        if read.tool != "get_team_leave" and (read.employee or "self").casefold() in _SELF_WORDS:
            read.employee = target


def _guard(plan: Plan, turn: Turn) -> None:
    if plan.intent == "manipulation":
        raise _Stop("refuse", MANIPULATION)
    if plan.intent == "off_topic":
        raise _Stop("refuse", OFF_TOPIC)
    if plan.intent == "greeting" and not plan.reads and not plan.policy_query:
        raise _Stop("clarify", GREETING)
    others = {r.employee for r in plan.reads if r.tool != "get_team_leave" and (r.employee or "self").casefold() not in _SELF_WORDS}
    if plan.many_employees and not all(r.tool == "get_team_leave" for r in plan.reads):
        raise _Stop("refuse", BULK)
    if len(others) > 1:
        raise _Stop("refuse", BULK)  # at most one other employee per message (§7.3)
    if plan.missing:
        raise _Stop("clarify", "To answer, I need: " + ", ".join(plan.missing) + ".")


def handle_turn(turn: Turn, request_id: str) -> TurnResult:
    hidden = _hidden_names(turn)

    def stop(route: str, answer: str) -> TurnResult:
        return TurnResult(finalize(request_id, answer, route, hidden_names=hidden))

    def written(outcome: WriteOutcome) -> TurnResult:
        return TurnResult(
            finalize(request_id, outcome.answer, outcome.route, (), outcome.tool_calls, outcome.pending, hidden)
        )

    ctx = ToolContext(turn.requester, turn.conn, turn.today, CalendarHolidays(turn.cache_db, turn.requester, turn.today))
    proposal = get_proposal(turn.conn, turn.session_id)
    if is_clear_confirmation(turn.message):  # decided in code, no LLM needed
        return written(execute(ctx, turn.session_id)) if proposal else stop("clarify", NOTHING_PENDING)

    if turn.llm is None:
        return stop("no_answer", UNAVAILABLE)
    history = recent_messages(turn.conn, turn.session_id, HISTORY_TURNS)
    plan = plan_message(
        turn.llm, turn.message, turn.requester, turn.today, history, proposal.summary if proposal else None, turn.deadline
    )
    if plan is None:
        return stop("no_answer", UNAVAILABLE)

    # Write actions (§8.2). Never executed without a stored proposal and a clear yes.
    if plan.intent == "confirm":
        if proposal is None:
            return stop("clarify", NOTHING_PENDING)
        if looks_like_confirmation(turn.message):
            return written(execute(ctx, turn.session_id))
        return stop("clarify", UNCLEAR_CONFIRMATION)  # the proposal stays
    if plan.intent in ("leave_change", "modify"):
        if plan.change is None:
            return stop("clarify", "Would you like to apply for leave or cancel a request, and for which dates?")
        previous = proposal if plan.intent == "modify" else None
        return written(propose(plan.change, ctx, turn.session_id, turn.today, previous))
    if proposal is not None:
        delete_proposal(turn.conn, turn.session_id)  # anything else discards it

    try:
        _pin_mentioned_ids(plan, turn)
        _guard(plan, turn)
        planned_calls = [(read.tool, _tool_args(turn, read)) for read in plan.reads]
    except _Stop as s:
        return stop(s.route, s.answer)

    # Read tools: each runs its own permission check again (§6.1).
    tool_calls: list[ToolCall] = []
    tool_results: list[dict[str, Any]] = []
    for name, args in planned_calls:
        result = TOOLS[name].run(ctx, args)
        if result.code == authz.ACCESS_DENIED:
            return stop("refuse", authz.ACCESS_DENIED_MESSAGE)
        tool_calls.append(ToolCall(name=name, args=args, status=result.status))
        tool_results.append({"tool": name, "args": args, "status": result.status, "data": result.data, "error": result.reason})

    # Documents.
    retrieval, segments, sent_ids, reference = None, [], [], turn.today
    wants_docs = plan.intent == "policy_question" or bool(plan.policy_query)
    if wants_docs and turn.retriever is not None:
        reference = resolve_reference_date(plan.about_date, turn.today) or turn.today
        retrieval = turn.retriever.retrieve(plan.policy_query or turn.message, turn.requester, reference, today=turn.today)
        segments, packed = build_corpus_input(retrieval, turn.retriever.corpus)
        sent_ids = list(packed.chunk_ids) if packed else []

    if not tool_calls and not sent_ids:
        return stop("no_answer", NO_ANSWER)

    composed = compose(
        turn.llm, turn.message, turn.requester, turn.today, reference, segments, tool_results, [], turn.deadline
    )
    corpus_tokens = sum(count_tokens(s.text) for s in segments) if composed is not None else 0
    if composed is None or (not composed.answer.strip() and composed.covered != "none"):
        return stop("no_answer", UNAVAILABLE)  # never send an empty answer
    used = grounded_chunk_ids(composed.used_chunk_ids, composed.answer, retrieval.resolutions if retrieval else ())
    citations: list[Citation] = build_citations(used, sent_ids, turn.retriever.corpus) if turn.retriever else []
    if wants_docs and not citations and not tool_calls:
        # A policy answer with no supported citation is a guess: don't send it (§8.1).
        return TurnResult(finalize(request_id, NO_ANSWER, "no_answer", hidden_names=hidden), corpus_tokens)
    if composed.covered == "none" or not composed.answer.strip():
        return TurnResult(finalize(request_id, NO_ANSWER, "no_answer", hidden_names=hidden), corpus_tokens)
    route = route_for(citations, tool_calls)
    return TurnResult(
        finalize(request_id, composed.answer, route, citations, tool_calls, hidden_names=hidden), corpus_tokens
    )
