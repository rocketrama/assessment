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
from app.orchestration import tool_answers
from app.orchestration.dates import (
    DateError,
    find_date_phrases,
    find_date_range,
    find_month_period,
    resolve_range,
    resolve_reference_date,
)
from app.orchestration.future_balance import (
    ACCRUAL_QUERY,
    asks_future_balance,
    calendar_note,
    credit_day,
    find_target_date,
)
from app.orchestration.planner import Plan, PlannedRead, plan_message
from app.orchestration.proposals import (
    ASK_CHANGE,
    NOTHING_PENDING,
    UNCLEAR_CONFIRMATION,
    WriteOutcome,
    execute,
    is_clear_confirmation,
    looks_like_confirmation,
    propose,
)
from app.orchestration.validator import (
    build_citations,
    finalize,
    grounded_chunk_ids,
    route_for,
    strip_internal_ids,
    superseded_values,
)
from app.retrieval.precedence import HIGH_AUTHORITY
from app.retrieval.retriever import Retriever
from app.tokens import count_tokens
from app.tools import authz
from app.tools.registry import TOOLS, ToolContext

log = logging.getLogger("veltrix.orchestration")

HISTORY_TURNS = 6
_EMPLOYEE_ID_RE = re.compile(r"^[A-Za-z]{1,3}\d{2,8}$")
# Deterministic bulk check (§7.3), in case the planner misses it: "everyone's balances",
# "all employees", "every employee"… A manager's own team is handled by get_team_leave.
_BULK_RE = re.compile(
    r"\b(every\s*one|every\s*body|every\s+employee|all\s+(?:the\s+)?(?:employees|staff|people|managers)|"
    r"each\s+employee|whole\s+(?:company|organi[sz]ation))",
    re.IGNORECASE,
)
_PEOPLE_DATA_RE = re.compile(r"\b(balances?|requests?|on leave|days left|leave data|leave records?)\b", re.IGNORECASE)
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
REWRITE_NOTE = (
    "Your previous answer was rejected: it states values from superseded or lower-precedence sources "
    "({values}). Rewrite it with the current values only. Keep an older value only if the employee asked how "
    "the rule changed or what an older version said. Previous answer: {answer}"
)
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


def _nothing_to_book(plan: Plan, message: str) -> bool:
    """A leave_change plan without a leave type, dates or request id, for a message that
    names no date either: there is nothing to propose yet."""
    change = plan.change
    given = change is not None and any((change.leave_type, change.start, change.end, change.request_id))
    return not given and not find_date_phrases(message)


def _hidden_names(turn: Turn) -> list[str]:
    """Full names the requester may not see; scrubbed from the answer (§7.4)."""
    rows = turn.conn.execute("SELECT employee_id, full_name FROM employees").fetchall()
    return [r["full_name"] for r in rows if not authz.can_view_employee(turn.conn, turn.requester, r["employee_id"])]


def _resolve_employee(turn: Turn, ref: str | None) -> str:
    """Planner's employee reference → an employee id the requester may see, or _Stop.
    Not visible and not existing give the same fixed refusal (§7.3)."""
    text = re.sub(r"['’]s$", "", (ref or "self").strip()).strip()  # "Test Employee Us's" → the name
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


def _ensure_future_balance(plan: Plan, turn: Turn) -> None:
    """A future-balance question needs the current balance and the accrual rules, whatever the planner chose."""
    if plan.intent not in ("policy_question", "leave_info") or not asks_future_balance(turn.message, turn.today):
        return
    if not any(r.tool == "get_leave_balance" for r in plan.reads):
        plan.reads.append(PlannedRead(tool="get_leave_balance", employee="self"))
    plan.intent = "policy_question"
    plan.policy_query = f"{plan.policy_query or turn.message} {ACCRUAL_QUERY}"


def _tool_args(turn: Turn, read: PlannedRead) -> dict[str, Any]:
    if read.tool == "get_team_leave":
        if not authz.can_view_team(turn.requester):
            raise _Stop("refuse", NOT_A_MANAGER)
        written = find_date_range(turn.message)
        found = find_date_phrases(turn.message)
        if not written and not found and (month := find_month_period(turn.message, turn.today)):
            return {"start_date": month.start.isoformat(), "end_date": month.end.isoformat()}
        start, end = written or (read.start, read.end)  # the message wins
        if not start:  # the planner dropped dates the user gave: take them from the message
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


def _named_employee_ids(turn: Turn) -> set[str]:
    """Ids of employees whose full name is written in the message. Longest names first,
    so "Test Employee In Two" is not also read as "Test Employee In"."""
    text = turn.message.casefold()
    found: set[str] = set()
    rows = turn.conn.execute("SELECT employee_id, full_name FROM employees").fetchall()
    for row in sorted(rows, key=lambda r: len(r["full_name"]), reverse=True):
        pattern = rf"\b{re.escape(row['full_name'].casefold())}\b"
        if re.search(pattern, text):
            found.add(row["employee_id"])
            text = re.sub(pattern, " ", text)
    return found


def _pin_mentioned_ids(plan: Plan, turn: Turn) -> None:
    """An employee id or full name written in the message beats the planner's "self":
    the live model once read "What is E1003's balance?" as the requester's own balance,
    and later "What is Test Employee Us's leave balance?" too."""
    written_ids = {m.upper() for m in _EMPLOYEE_ID_IN_TEXT_RE.findall(turn.message)}
    mentioned = (written_ids | _named_employee_ids(turn)) - {turn.requester.employee_id}
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
    # Eval finding: the planner called "Show me every employee's leave balance" a policy question.
    wants_people_data = plan.intent == "leave_info" or bool(plan.reads) or _PEOPLE_DATA_RE.search(turn.message) is not None
    bulk = plan.many_employees or (wants_people_data and _BULK_RE.search(turn.message) is not None)
    only_team_leave = bool(plan.reads) and all(r.tool == "get_team_leave" for r in plan.reads)
    if bulk and not only_team_leave:
        raise _Stop("refuse", BULK)
    if len(others) > 1:
        raise _Stop("refuse", BULK)  # at most one other employee per message (§7.3)
    # Eval finding: the planner listed "leave type" as missing for a balance lookup. With a
    # lookup or a search to run, missing details are checked where they matter (dates in
    # _tool_args, write details in proposals), not taken from the plan.
    if plan.missing and not plan.reads and not plan.policy_query:
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
    if plan.intent == "leave_change" and plan.policy_query and _nothing_to_book(plan, turn.message):
        # Eval finding: "My grandmother passed away, how long can I stay home?" was planned
        # as leave_change in 6 of 11 replies, with a fitting policy_query and nothing to book.
        # A booking needs details; a question needs an answer, so answer it.
        log.info("planner intent corrected", extra={"request_id": request_id, "reason": "question, not a booking"})
        plan = plan.model_copy(update={"intent": "policy_question", "change": None, "missing": []})
    if plan.intent in ("leave_change", "modify"):
        if plan.change is None:
            return stop("clarify", ASK_CHANGE)
        previous = proposal if plan.intent == "modify" else None
        return written(propose(plan.change, ctx, turn.session_id, turn.today, previous, turn.message))
    if proposal is not None:
        delete_proposal(turn.conn, turn.session_id)  # anything else discards it

    try:
        _pin_mentioned_ids(plan, turn)
        _guard(plan, turn)
        _ensure_future_balance(plan, turn)
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

    wants_docs = plan.intent == "policy_question" or bool(plan.policy_query)
    if not wants_docs and tool_answers.can_render(tool_results):
        # Records only: code writes the answer, every record, no LLM call.
        answer = tool_answers.render(tool_results, turn.requester.employee_id)
        return TurnResult(finalize(request_id, answer, "tool", (), tool_calls, hidden_names=hidden))

    # Documents.
    retrieval, segments, sent_ids, reference = None, [], [], turn.today
    if wants_docs and turn.retriever is not None:
        reference = resolve_reference_date(plan.about_date, turn.today) or turn.today
        retrieval = turn.retriever.retrieve(
            plan.policy_query or turn.message, turn.requester, reference, today=turn.today, message=turn.message
        )
        segments, packed = build_corpus_input(retrieval, turn.retriever.corpus)
        sent_ids = list(packed.chunk_ids) if packed else []

    if not tool_calls and not sent_ids:
        log.info("no_answer", extra={"reason": "nothing retrieved or looked up"})
        return stop("no_answer", NO_ANSWER)

    notes: list[str] = []
    has_balance = any(c.name == "get_leave_balance" and c.status == "success" for c in tool_calls)
    if wants_docs and has_balance and (target := find_target_date(turn.message, turn.today)):
        eligible_facts = [f for r in retrieval.resolutions for f in r.winners] if retrieval else []
        notes.append(calendar_note(turn.today, target, turn.requester.join_date, credit_day(eligible_facts)))

    def write(extra_notes: list[str]):
        return compose(
            turn.llm, turn.message, turn.requester, turn.today, reference, segments, tool_results,
            notes + extra_notes, turn.deadline,
        )

    composed = write([])
    corpus_tokens = sum(count_tokens(s.text) for s in segments) if composed is not None else 0
    if composed is None or (not composed.answer.strip() and composed.covered != "none"):
        return stop("no_answer", UNAVAILABLE)  # never send an empty answer
    # Answer check: superseded values go back to the composer once, with the exact values
    # found; without time for that (or a usable rewrite), the first answer stands.
    superseded = superseded_values(composed.answer, retrieval.resolutions) if retrieval else []
    if superseded and composed.covered != "none":
        log.info("answer_check", extra={"reason": "superseded values", "values": superseded})
        retry = write([REWRITE_NOTE.format(values="; ".join(superseded), answer=composed.answer)])
        if retry is not None and retry.answer.strip() and retry.covered != "none":
            composed = retry
    if turn.retriever is not None:
        composed = composed.model_copy(
            update={"answer": strip_internal_ids(composed.answer, turn.retriever.corpus.docs)}
        )
    low_authority: dict[str, list[str]] = {}
    if turn.retriever is not None:
        corpus = turn.retriever.corpus
        low_authority = {
            c: [f.value for f in corpus.facts_by_chunk.get(c, [])]
            for c in sent_ids
            if c in corpus.chunks and corpus.docs[corpus.chunks[c].doc_id].doc_type not in HIGH_AUTHORITY
        }
    used = grounded_chunk_ids(
        composed.used_chunk_ids, composed.answer, retrieval.resolutions if retrieval else (), low_authority
    )
    # Rule-table rows are sent as facts, not passages: their chunks count as sent.
    rule_chunks = [f.chunk_id for r in retrieval.resolutions for f in r.winners + r.upcoming] if retrieval else []
    citations: list[Citation] = (
        build_citations(used, list(sent_ids) + rule_chunks, turn.retriever.corpus) if turn.retriever else []
    )
    if wants_docs and not citations and not tool_calls:
        # A policy answer with no supported citation is a guess: don't send it (§8.1).
        log.info(
            "no_answer",
            extra={"reason": "no supported citation", "covered": composed.covered, "used": len(composed.used_chunk_ids)},
        )
        return TurnResult(finalize(request_id, NO_ANSWER, "no_answer", hidden_names=hidden), corpus_tokens)
    if composed.covered == "none" or not composed.answer.strip():
        log.info("no_answer", extra={"reason": "composer covered none"})
        return TurnResult(finalize(request_id, NO_ANSWER, "no_answer", hidden_names=hidden), corpus_tokens)
    route = route_for(citations, tool_calls)
    return TurnResult(
        finalize(request_id, composed.answer, route, citations, tool_calls, hidden_names=hidden), corpus_tokens
    )
