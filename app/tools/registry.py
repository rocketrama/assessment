"""The five leave tools (TASK.pdf §6.1), with exactly these tool and argument names.

Every tool gets a `ToolContext` built from the X-Employee-Id header, and runs its
permission check (tools/authz.py) itself; LLM-chosen arguments never decide who
the requester is. Results carry IDs, dates and numbers only — no names, emails
or phones.

Read tools have `run`. Write tools have two halves for the propose/confirm flow:
`validate` (dry run, for the proposal) and `execute`, which must run inside
`db.transaction()` and re-checks every rule under the lock before writing (§8.2),
so concurrent requests can't drive a balance negative or refund twice.
"""

import sqlite3
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.data.employees import Employee
from app.data.leave import (
    adjust_balance,
    get_balances,
    get_request,
    insert_request,
    list_requests,
    overlapping_requests,
    set_status,
    team_requests,
)
from app.tools import authz
from app.tools.leave_rules import (
    RuleViolation,
    check_apply_amount,
    check_apply_dates,
    check_cancel,
    check_leave_type,
    parse_date,
)
from app.tools.working_days import HolidaySource, count_working_days


@dataclass(frozen=True)
class ToolContext:
    requester: Employee
    conn: sqlite3.Connection
    today: date
    holidays: HolidaySource


@dataclass(frozen=True)
class ToolResult:
    status: Literal["success", "error"]
    data: dict[str, Any] = field(default_factory=dict)
    code: str | None = None
    reason: str | None = None

    @classmethod
    def error(cls, code: str, reason: str) -> "ToolResult":
        return cls("error", code=code, reason=reason)

    @classmethod
    def violation(cls, v: RuleViolation) -> "ToolResult":
        return cls.error(v.code, v.reason)


# --- argument models (strict: no "1001" → 1001 coercion, no unknown keys) ----

class _Args(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class GetLeaveBalanceArgs(_Args):
    employee_id: str = Field(min_length=1, max_length=64)
    leave_type: str | None = None


class ListLeaveRequestsArgs(_Args):
    employee_id: str = Field(min_length=1, max_length=64)
    status: str | None = None


class GetTeamLeaveArgs(_Args):
    start_date: str
    end_date: str


class ApplyLeaveArgs(_Args):
    leave_type: str
    start_date: str
    end_date: str


class CancelLeaveArgs(_Args):
    request_id: str = Field(min_length=1, max_length=64)


def _parse_args(model: type[_Args], args: dict[str, Any]) -> _Args | ToolResult:
    try:
        return model.model_validate(args)
    except ValidationError as exc:
        # Field names and error kinds only; the values themselves are not echoed.
        fields = ", ".join(f"{'.'.join(map(str, e['loc'])) or 'args'}: {e['msg']}" for e in exc.errors())
        return ToolResult.error("INVALID_ARGS", f"Invalid tool arguments ({fields}).")


def _parse_range(start_s: str, end_s: str) -> tuple[date, date] | ToolResult:
    start = parse_date(start_s, "start_date")
    if isinstance(start, RuleViolation):
        return ToolResult.violation(start)
    end = parse_date(end_s, "end_date")
    if isinstance(end, RuleViolation):
        return ToolResult.violation(end)
    return start, end


def _requests_data(requests: list) -> list[dict[str, Any]]:
    return [asdict(r) for r in requests]


# --- read tools -------------------------------------------------------------

def get_leave_balance(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    a = _parse_args(GetLeaveBalanceArgs, args)
    if isinstance(a, ToolResult):
        return a
    if a.leave_type is not None and (v := check_leave_type(a.leave_type)):
        return ToolResult.violation(v)
    if not authz.can_view_employee(ctx.conn, ctx.requester, a.employee_id):
        return ToolResult.error(authz.ACCESS_DENIED, authz.ACCESS_DENIED_MESSAGE)
    balances = get_balances(ctx.conn, a.employee_id, a.leave_type)
    return ToolResult(
        "success",
        {
            "employee_id": a.employee_id,
            "balances": [{"leave_type": b.leave_type, "balance_days": b.balance_days} for b in balances],
        },
    )


def list_leave_requests(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    a = _parse_args(ListLeaveRequestsArgs, args)
    if isinstance(a, ToolResult):
        return a
    if not authz.can_view_employee(ctx.conn, ctx.requester, a.employee_id):
        return ToolResult.error(authz.ACCESS_DENIED, authz.ACCESS_DENIED_MESSAGE)
    requests = list_requests(ctx.conn, a.employee_id, a.status)
    return ToolResult("success", {"employee_id": a.employee_id, "requests": _requests_data(requests)})


def get_team_leave(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    a = _parse_args(GetTeamLeaveArgs, args)
    if isinstance(a, ToolResult):
        return a
    if not authz.can_view_team(ctx.requester):
        return ToolResult.error("NOT_A_MANAGER", "Only managers can view their team's leave.")
    parsed = _parse_range(a.start_date, a.end_date)
    if isinstance(parsed, ToolResult):
        return parsed
    start, end = parsed
    if end < start:
        return ToolResult.error("END_BEFORE_START", "The end date is before the start date.")
    requests = team_requests(ctx.conn, ctx.requester.employee_id, start.isoformat(), end.isoformat())
    return ToolResult(
        "success",
        {"start_date": a.start_date, "end_date": a.end_date, "requests": _requests_data(requests)},
    )


# --- apply_leave ------------------------------------------------------------

def _check_apply(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any] | ToolResult:
    """Every §6.2 rule against current data; returns the request details or an error."""
    a = _parse_args(ApplyLeaveArgs, args)
    if isinstance(a, ToolResult):
        return a
    if v := check_leave_type(a.leave_type):
        return ToolResult.violation(v)
    parsed = _parse_range(a.start_date, a.end_date)
    if isinstance(parsed, ToolResult):
        return parsed
    start, end = parsed
    if v := check_apply_dates(start, end, ctx.today):
        return ToolResult.violation(v)

    me = ctx.requester.employee_id
    holidays = ctx.holidays.holidays(ctx.requester.region, start, end)
    working_days = count_working_days(start, end, holidays)
    balances = get_balances(ctx.conn, me, a.leave_type)
    balance = balances[0].balance_days if balances else None
    overlaps = overlapping_requests(ctx.conn, me, start.isoformat(), end.isoformat())
    if v := check_apply_amount(a.leave_type, working_days, balance, overlaps):
        return ToolResult.violation(v)
    return {
        "employee_id": me,
        "leave_type": a.leave_type,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "working_days": working_days,
        "balance_days": balance,
        "balance_after": balance - working_days,
    }


def validate_apply_leave(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    checked = _check_apply(ctx, args)
    return checked if isinstance(checked, ToolResult) else ToolResult("success", checked)


def execute_apply_leave(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    _require_transaction(ctx.conn)
    checked = _check_apply(ctx, args)
    if isinstance(checked, ToolResult):
        return checked
    request = insert_request(
        ctx.conn,
        checked["employee_id"],
        checked["leave_type"],
        checked["start_date"],
        checked["end_date"],
        checked["working_days"],
    )
    new_balance = adjust_balance(
        ctx.conn, checked["employee_id"], checked["leave_type"], -checked["working_days"]
    )
    return ToolResult("success", {**asdict(request), "balance_after": new_balance})


# --- cancel_leave -----------------------------------------------------------

def _check_cancel(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any] | ToolResult:
    a = _parse_args(CancelLeaveArgs, args)
    if isinstance(a, ToolResult):
        return a
    request = get_request(ctx.conn, a.request_id)
    if not authz.owns_request(ctx.requester, request):
        return ToolResult.error(authz.REQUEST_NOT_FOUND, authz.REQUEST_NOT_FOUND_MESSAGE)
    if v := check_cancel(request, ctx.today):
        return ToolResult.violation(v)
    return asdict(request)


def validate_cancel_leave(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    checked = _check_cancel(ctx, args)
    return checked if isinstance(checked, ToolResult) else ToolResult("success", checked)


def execute_cancel_leave(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    _require_transaction(ctx.conn)
    checked = _check_cancel(ctx, args)
    if isinstance(checked, ToolResult):
        return checked
    set_status(ctx.conn, checked["request_id"], "cancelled")
    new_balance = adjust_balance(
        ctx.conn, checked["employee_id"], checked["leave_type"], checked["working_days"]
    )
    return ToolResult("success", {**checked, "status": "cancelled", "balance_after": new_balance})


def _require_transaction(conn: sqlite3.Connection) -> None:
    if not conn.in_transaction:
        raise RuntimeError("write tools must execute inside db.transaction()")


# --- registry ---------------------------------------------------------------

ToolFn = Callable[[ToolContext, dict[str, Any]], ToolResult]


@dataclass(frozen=True)
class Tool:
    name: str
    args_model: type[_Args]
    run: ToolFn | None = None
    validate: ToolFn | None = None
    execute: ToolFn | None = None

    @property
    def writes(self) -> bool:
        return self.execute is not None


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in (
        Tool("get_leave_balance", GetLeaveBalanceArgs, run=get_leave_balance),
        Tool("list_leave_requests", ListLeaveRequestsArgs, run=list_leave_requests),
        Tool("get_team_leave", GetTeamLeaveArgs, run=get_team_leave),
        Tool("apply_leave", ApplyLeaveArgs, validate=validate_apply_leave, execute=execute_apply_leave),
        Tool("cancel_leave", CancelLeaveArgs, validate=validate_cancel_leave, execute=execute_cancel_leave),
    )
}
