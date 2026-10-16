"""apply_leave and cancel_leave rules (TASK.pdf §6.2, §6.3). Pure functions, no DB.

Each check returns None when the action is allowed, or a `RuleViolation` whose
`reason` is the specific sentence the answer must give (§6.4).
"""

from dataclasses import dataclass
from datetime import date

from app.data.leave import ACTIVE_STATUSES, LeaveRequest

LEAVE_TYPES = ("annual", "sick")
LEAVE_YEAR = 2026


@dataclass(frozen=True)
class RuleViolation:
    code: str
    reason: str


def parse_date(value: str, field: str) -> date | RuleViolation:
    # Length check: fromisoformat also accepts compact forms like 20261102.
    if len(value) == 10:
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    return RuleViolation("INVALID_DATE", f"{field} must be a real date in YYYY-MM-DD form.")


def check_leave_type(leave_type: str) -> RuleViolation | None:
    if leave_type not in LEAVE_TYPES:
        return RuleViolation("INVALID_LEAVE_TYPE", "Leave type must be annual or sick.")
    return None


def check_apply_dates(start: date, end: date, today: date) -> RuleViolation | None:
    if start.year != LEAVE_YEAR or end.year != LEAVE_YEAR:
        return RuleViolation("OUTSIDE_LEAVE_YEAR", f"Leave can only be requested for dates in {LEAVE_YEAR}.")
    if start < today:
        return RuleViolation(
            "START_IN_PAST", f"The start date {start.isoformat()} is before today ({today.isoformat()})."
        )
    if end < start:
        return RuleViolation(
            "END_BEFORE_START",
            f"The end date {end.isoformat()} is before the start date {start.isoformat()}.",
        )
    return None


def check_apply_amount(
    leave_type: str,
    working_days: int,
    balance_days: float | None,
    overlapping: list[LeaveRequest],
) -> RuleViolation | None:
    if working_days == 0:
        return RuleViolation(
            "NO_WORKING_DAYS", "The requested dates contain no working days (only weekends or public holidays)."
        )
    if balance_days is None:
        return RuleViolation("NO_BALANCE", f"There is no {leave_type} leave balance on record.")
    if working_days > balance_days:
        return RuleViolation(
            "INSUFFICIENT_BALANCE",
            f"The request needs {working_days} working days of {leave_type} leave, "
            f"but the balance is {balance_days:g} days.",
        )
    if overlapping:
        r = overlapping[0]
        return RuleViolation(
            "OVERLAP",
            f"The dates overlap your {r.status} {r.leave_type} leave request {r.request_id} "
            f"({r.start_date} to {r.end_date}).",
        )
    return None


def check_cancel(request: LeaveRequest, today: date) -> RuleViolation | None:
    if request.status not in ACTIVE_STATUSES:
        return RuleViolation(
            "NOT_CANCELLABLE_STATUS",
            f"Request {request.request_id} is {request.status}; only pending or approved requests can be cancelled.",
        )
    if date.fromisoformat(request.start_date) <= today:
        return RuleViolation(
            "ALREADY_STARTED",
            f"Request {request.request_id} starts on {request.start_date}; only leave starting after today "
            f"({today.isoformat()}) can be cancelled.",
        )
    return None
