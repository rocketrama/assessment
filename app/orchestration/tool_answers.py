"""Answers for tool-only turns, written in code (no composer call).

Balances, requests and team leave are records: the answer is the records, so
code writes it. Eval finding: the composer sometimes left the pending request
out of a team-leave answer although the tool returned it. Code lists every
record, the same way every time, and saves an LLM call.

Only IDs, dates, numbers and statuses appear, as in the tool results (§7.4).
"""

from datetime import date
from typing import Any


def _day(d: str) -> str:
    return date.fromisoformat(d).strftime("%A %d %B %Y").replace(" 0", " ")


def _span(start: str, end: str) -> str:
    return _day(start) if start == end else f"{_day(start)} to {_day(end)}"


def _days(n: float) -> str:
    return f"{n:g} day{'' if n == 1 else 's'}"


def _whose(employee_id: str, requester_id: str) -> str:
    return "Your" if employee_id == requester_id else f"{employee_id}'s"


def _request_line(r: dict[str, Any], with_employee: bool = False) -> str:
    who = f"{r['employee_id']}: " if with_employee else ""
    days = f"{r['working_days']:g} working day{'' if r['working_days'] == 1 else 's'}"
    return (
        f"- {who}{r['leave_type']} leave, {_span(r['start_date'], r['end_date'])} ({days}), "
        f"{r['status']} ({r['request_id']})"
    )


def _balance(result: dict[str, Any], requester_id: str) -> str:
    data = result["data"]
    whose = _whose(data["employee_id"], requester_id)
    balances = data["balances"]
    if not balances:
        kind = f"{result['args']['leave_type']} leave " if result["args"].get("leave_type") else ""
        return f"No {kind}balance was found for {'you' if whose == 'Your' else data['employee_id']}."
    if len(balances) == 1:
        b = balances[0]
        return f"{whose} {b['leave_type']} leave balance is {_days(b['balance_days'])}."
    parts = ", ".join(f"{b['leave_type']} {_days(b['balance_days'])}" for b in balances)
    return f"{whose} leave balances: {parts}."


def _requests(result: dict[str, Any], requester_id: str) -> str:
    data = result["data"]
    whose = _whose(data["employee_id"], requester_id)
    status = result["args"].get("status")
    kind = f"{status} " if status else ""
    if not data["requests"]:
        subject = "You have" if whose == "Your" else f"{data['employee_id']} has"
        return f"{subject} no {kind}leave requests."
    lines = [f"{whose} {kind}leave requests:"] + [_request_line(r) for r in data["requests"]]
    return "\n".join(lines)


def _team(result: dict[str, Any]) -> str:
    data = result["data"]
    span = _span(data["start_date"], data["end_date"])
    requests = data["requests"]
    if not requests:
        return f"None of your direct reports has pending or approved leave on {span}."
    people = len({r["employee_id"] for r in requests})
    head = (
        f"{people} of your direct reports {'has' if people == 1 else 'have'} pending or approved leave "
        f"overlapping {span}:"
    )
    return "\n".join([head] + [_request_line(r, with_employee=True) for r in requests])


_RENDERERS = {
    "get_leave_balance": _balance,
    "list_leave_requests": _requests,
    "get_team_leave": lambda result, _requester: _team(result),
}


def can_render(tool_results: list[dict[str, Any]]) -> bool:
    return bool(tool_results) and all(r["tool"] in _RENDERERS and r["status"] == "success" for r in tool_results)


def render(tool_results: list[dict[str, Any]], requester_id: str) -> str:
    return "\n\n".join(_RENDERERS[r["tool"]](r, requester_id) for r in tool_results)
