"""apply_leave / cancel_leave rules (TASK.pdf §6.2, §6.3): every rejection reason."""

from datetime import date

import pytest

from app.data.leave import LeaveRequest
from app.tools.leave_rules import (
    RuleViolation,
    check_apply_amount,
    check_apply_dates,
    check_cancel,
    check_leave_type,
    parse_date,
)

D = date.fromisoformat
TODAY = D("2026-09-15")


def _req(status="approved", start="2026-10-05", end="2026-10-07"):
    return LeaveRequest("R1", "E1001", "annual", start, end, 3, status, "2026-08-01T00:00:00Z")


def _code(result):
    return result.code if isinstance(result, RuleViolation) else None


@pytest.mark.parametrize("value", ["2026-02-30", "20261102", "tomorrow", "", "2026-1-5"])
def test_parse_date_rejects_non_iso(value):
    assert _code(parse_date(value, "start_date")) == "INVALID_DATE"


def test_parse_date_accepts_iso():
    assert parse_date("2026-11-02", "start_date") == D("2026-11-02")


@pytest.mark.parametrize("leave_type, code", [("annual", None), ("sick", None), ("casual", "INVALID_LEAVE_TYPE"), ("Annual", "INVALID_LEAVE_TYPE")])
def test_leave_type(leave_type, code):
    assert _code(check_leave_type(leave_type)) == code


@pytest.mark.parametrize(
    "start, end, code",
    [
        ("2026-11-02", "2026-11-06", None),
        ("2026-09-15", "2026-09-15", None),  # start = today is allowed
        ("2026-09-14", "2026-09-16", "START_IN_PAST"),
        ("2026-12-30", "2027-01-04", "OUTSIDE_LEAVE_YEAR"),
        ("2025-12-29", "2026-01-02", "OUTSIDE_LEAVE_YEAR"),
        ("2026-11-06", "2026-11-02", "END_BEFORE_START"),
    ],
)
def test_apply_dates(start, end, code):
    assert _code(check_apply_dates(D(start), D(end), TODAY)) == code


@pytest.mark.parametrize(
    "working_days, balance, overlapping, code",
    [
        (3, 12.0, [], None),
        (12, 12.0, [], None),  # exactly the balance
        (0, 12.0, [], "NO_WORKING_DAYS"),
        (3, None, [], "NO_BALANCE"),
        (13, 12.0, [], "INSUFFICIENT_BALANCE"),
        (3, 2.5, [], "INSUFFICIENT_BALANCE"),
        (3, 12.0, [_req()], "OVERLAP"),
    ],
)
def test_apply_amount(working_days, balance, overlapping, code):
    assert _code(check_apply_amount("annual", working_days, balance, overlapping)) == code


def test_rejection_reasons_are_specific():
    v = check_apply_amount("sick", 9, 6.0, [])
    assert v.reason == "The request needs 9 working days of sick leave, but the balance is 6 days."
    v = check_apply_amount("annual", 3, 12.0, [_req()])
    assert "R1" in v.reason and "2026-10-05 to 2026-10-07" in v.reason


@pytest.mark.parametrize(
    "status, start, code",
    [
        ("approved", "2026-10-05", None),
        ("pending", "2026-09-16", None),  # tomorrow
        ("approved", "2026-09-15", "ALREADY_STARTED"),  # today is not "after today"
        ("approved", "2026-08-03", "ALREADY_STARTED"),
        ("cancelled", "2026-10-05", "NOT_CANCELLABLE_STATUS"),
        ("rejected", "2026-10-05", "NOT_CANCELLABLE_STATUS"),
    ],
)
def test_cancel(status, start, code):
    assert _code(check_cancel(_req(status=status, start=start), TODAY)) == code
