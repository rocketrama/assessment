"""Write actions: propose → confirm → execute (TASK.pdf §8.2, ARCHITECTURE §10).

- Details are collected without guessing: the leave type is never assumed, dates
  go through dates.py, and everything missing is asked for in one message.
- Proposing runs every §6.2/§6.3 rule as a dry run. A violation is reported at
  once (route tool, status error); otherwise the arguments are stored
  server-side and the user is asked to confirm. Nothing is written yet.
- Only a clear confirmation executes, and it executes the *stored* arguments:
  the proposal is claimed and the leave written in one BEGIN IMMEDIATE
  transaction, where every rule is checked again, so a repeated "yes" or a
  concurrent request can never apply leave twice.
- Messages here are fixed text built from tool results — no LLM writes them —
  so summaries are exact and contain IDs, dates and numbers only.
"""

import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from app.api.schemas import ToolCall
from app.data.db import transaction
from app.data.leave import ACTIVE_STATUSES, LeaveRequest, list_requests
from app.data.sessions import Proposal, save_proposal, take_proposal
from app.orchestration.dates import DateError, resolve_date, resolve_end, resolve_range
from app.orchestration.planner import PlannedChange
from app.tools.registry import TOOLS, ToolContext

NOTHING_PENDING = "There's nothing waiting for your confirmation."
CONFIRM_PROMPT = "Shall I go ahead? Reply \"yes\" to confirm, or tell me what to change."
UNCLEAR_CONFIRMATION = "Please reply \"yes\" to confirm, or tell me what to change."

_CONFIRM_WORDS = (
    "yes", "y", "yeah", "yep", "yup", "sure", "ok", "okay", "confirm", "confirmed", "go ahead", "go for it",
    "do it", "please do", "proceed", "approve", "sounds good", "that's fine", "thats fine", "correct",
)
_CONFIRM_RE = re.compile(
    r"^(?:(?:" + "|".join(re.escape(w) for w in _CONFIRM_WORDS) + r")[\s,.!]*)+"
    r"(?:please|thanks|thank you|go ahead|do it)?[\s.!]*$"
)
_NEGATION_RE = re.compile(r"\b(no|not|don'?t|wait|stop|instead|but|change|cancel|until|from|make)\b")


def is_clear_confirmation(message: str) -> bool:
    """Deterministic: a short message made only of agreement words."""
    return bool(_CONFIRM_RE.match(" ".join(message.lower().split())))


def looks_like_confirmation(message: str) -> bool:
    """For a planner 'confirm': short and with no word that changes or rejects."""
    text = message.lower()
    return len(text.split()) <= 6 and not _NEGATION_RE.search(text)


@dataclass
class WriteOutcome:
    answer: str
    route: str  # "tool" | "clarify"
    tool_calls: list[ToolCall] = field(default_factory=list)
    pending: bool = False


def _long(d: str) -> str:
    return date.fromisoformat(d).strftime("%A %d %B %Y").replace(" 0", " ")


def _days(n: float) -> str:
    return f"{n:g} working day{'s' if n != 1 else ''}"


# --- collecting details -------------------------------------------------------

def _apply_args(change: PlannedChange, base: dict[str, Any] | None, today: date) -> dict[str, Any] | WriteOutcome:
    """leave_type / start_date / end_date from the change, on top of an earlier
    proposal when modifying. Cases:
    - new start only: one day, or Mon–Fri for "next week"; when modifying a
      single-day start, the old end is kept if it is still on or after it;
    - end only ("make it until Friday"): the start is kept, the end moves;
    - both: as given."""
    base = base or {}
    leave_type = change.leave_type or base.get("leave_type")
    start = date.fromisoformat(base["start_date"]) if base.get("start_date") else None
    end = date.fromisoformat(base["end_date"]) if base.get("end_date") else None
    try:
        if change.start:
            span = resolve_range(change.start, None, today, allow_past=False)
            keep_old_end = end is not None and span.start == span.end and end >= span.start and not change.end
            start, end = span.start, (end if keep_old_end else span.end)
        if change.end and start is not None:
            end = resolve_end(change.end, start, today)
            if end < today:
                raise DateError(change.end, "past")
    except DateError as exc:
        reason = {"past": "has already passed", "ambiguous": "could mean more than one day"}.get(
            exc.reason, "isn't clear to me"
        )
        return WriteOutcome(f"\"{exc.phrase}\" {reason}. Which dates do you want?", "clarify")
    missing = []
    if leave_type not in ("annual", "sick"):
        missing.append("the leave type (annual or sick)")
    if start is None:
        missing.append("the dates")
    if missing:
        return WriteOutcome("To book leave I need " + " and ".join(missing) + ".", "clarify")
    return {"leave_type": leave_type, "start_date": start.isoformat(), "end_date": (end or start).isoformat()}


def _cancellable(conn: sqlite3.Connection, employee_id: str, today: date) -> list[LeaveRequest]:
    return [
        r for r in list_requests(conn, employee_id)
        if r.status in ACTIVE_STATUSES and date.fromisoformat(r.start_date) > today
    ]


def _describe(r: LeaveRequest) -> str:
    return f"{r.request_id} ({r.leave_type}, {r.start_date} to {r.end_date})"


def _cancel_args(
    change: PlannedChange, ctx: ToolContext, today: date
) -> dict[str, Any] | WriteOutcome:
    if change.request_id:
        return {"request_id": change.request_id.strip()}
    options = _cancellable(ctx.conn, ctx.requester.employee_id, today)
    if change.start:
        try:
            on = resolve_date(change.start, today)
        except DateError:
            on = None
        matches = [r for r in options if on and r.start_date <= on.isoformat() <= r.end_date]
        if len(matches) == 1:
            return {"request_id": matches[0].request_id}
        options = matches or options
    if len(options) == 1 and not change.start:
        return {"request_id": options[0].request_id}
    if not options:
        call = TOOLS["list_leave_requests"].run(ctx, {"employee_id": ctx.requester.employee_id})
        return WriteOutcome(
            "You have no pending or approved leave starting after today, so there is nothing to cancel.",
            "tool",
            [ToolCall(name="list_leave_requests", args={"employee_id": ctx.requester.employee_id}, status=call.status)],
        )
    listed = "; ".join(_describe(r) for r in options)
    return WriteOutcome(f"Which request should I cancel? {listed}.", "clarify")


# --- propose / execute ----------------------------------------------------------

def _summary(tool_name: str, data: dict[str, Any]) -> str:
    if tool_name == "apply_leave":
        span = _long(data["start_date"]) if data["start_date"] == data["end_date"] else (
            f"{_long(data['start_date'])} to {_long(data['end_date'])}"
        )
        return (
            f"{data['leave_type'].capitalize()} leave, {span}: {_days(data['working_days'])}. "
            f"Your {data['leave_type']} balance would go from {data['balance_days']:g} to {data['balance_after']:g} days."
        )
    return (
        f"Cancel request {data['request_id']} ({data['leave_type']} leave, {data['start_date']} to {data['end_date']}, "
        f"{_days(data['working_days'])}). The days go back to your {data['leave_type']} balance."
    )


def propose(
    change: PlannedChange,
    ctx: ToolContext,
    session_id: str,
    today: date,
    previous: Proposal | None = None,
) -> WriteOutcome:
    tool_name = "apply_leave" if change.action == "apply" else "cancel_leave"
    base = previous.args if previous and previous.tool_name == tool_name else None
    args = _apply_args(change, base, today) if tool_name == "apply_leave" else _cancel_args(change, ctx, today)
    if isinstance(args, WriteOutcome):
        return args
    result = TOOLS[tool_name].validate(ctx, args)
    if result.status == "error":
        return WriteOutcome(
            f"I can't {'book' if tool_name == 'apply_leave' else 'cancel'} that: {result.reason}",
            "tool",
            [ToolCall(name=tool_name, args=args, status="error")],
        )
    summary = _summary(tool_name, result.data)
    save_proposal(ctx.conn, session_id, ctx.requester.employee_id, tool_name, args, summary)
    return WriteOutcome(
        f"{summary} {CONFIRM_PROMPT}",
        "tool",
        [ToolCall(name=tool_name, args=args, status="pending_confirmation")],
        pending=True,
    )


def execute(ctx: ToolContext, session_id: str) -> WriteOutcome:
    """Claim the stored proposal and run it in one transaction (re-validated)."""
    with transaction(ctx.conn):
        proposal = take_proposal(ctx.conn, session_id)
        if proposal is None or proposal.employee_id != ctx.requester.employee_id:
            return WriteOutcome(NOTHING_PENDING, "clarify")
        result = TOOLS[proposal.tool_name].execute(ctx, proposal.args)
    call = [ToolCall(name=proposal.tool_name, args=proposal.args, status=result.status)]
    if result.status == "error":
        return WriteOutcome(f"I couldn't complete it: {result.reason}", "tool", call)
    d = result.data
    if proposal.tool_name == "apply_leave":
        answer = (
            f"Done. Request {d['request_id']} for {d['leave_type']} leave from {d['start_date']} to {d['end_date']} "
            f"({_days(d['working_days'])}) is pending approval. Your {d['leave_type']} balance is now "
            f"{d['balance_after']:g} days."
        )
    else:
        answer = (
            f"Done. Request {d['request_id']} is cancelled and {_days(d['working_days'])} went back to your "
            f"{d['leave_type']} balance, which is now {d['balance_after']:g} days."
        )
    return WriteOutcome(answer, "tool", call)
