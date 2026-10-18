"""Tool-only answers written in code, and the future-balance calendar."""

from datetime import date

import pytest

from app.orchestration import tool_answers
from app.orchestration.future_balance import asks_future_balance, calendar_note, credit_day, find_target_date

REQ = {"request_id": "REQ-1", "employee_id": "E1001", "leave_type": "annual", "start_date": "2026-10-05",
       "end_date": "2026-10-07", "working_days": 3, "status": "approved", "created_at": "x"}
PENDING = {**REQ, "request_id": "REQ-2", "employee_id": "E1002", "start_date": "2026-10-06",
           "end_date": "2026-10-09", "working_days": 4, "status": "pending"}


def result(tool, data, **args):
    return {"tool": tool, "args": args, "status": "success", "data": data, "error": None}


def test_own_single_balance():
    r = result("get_leave_balance", {"employee_id": "E1001", "balances": [{"leave_type": "sick", "balance_days": 6.0}]})
    assert tool_answers.render([r], "E1001") == "Your sick leave balance is 6 days."


def test_someone_elses_balances():
    r = result("get_leave_balance", {"employee_id": "E1002", "balances": [
        {"leave_type": "annual", "balance_days": 12.5}, {"leave_type": "sick", "balance_days": 1.0}]})
    assert tool_answers.render([r], "M2001") == "E1002's leave balances: annual 12.5 days, sick 1 day."


def test_team_leave_lists_every_request_pending_included():
    """Eval finding: the composer sometimes left out the pending request."""
    r = result("get_team_leave", {"start_date": "2026-10-05", "end_date": "2026-10-09", "requests": [REQ, PENDING]})
    text = tool_answers.render([r], "M2001")
    assert text.startswith("2 of your direct reports have pending or approved leave overlapping Monday 5 October 2026")
    assert "- E1001: annual leave, Monday 5 October 2026 to Wednesday 7 October 2026 (3 working days), approved (REQ-1)" in text
    assert "E1002" in text and "pending (REQ-2)" in text


def test_empty_results():
    team = result("get_team_leave", {"start_date": "2026-10-05", "end_date": "2026-10-05", "requests": []})
    assert tool_answers.render([team], "M2001") == (
        "None of your direct reports has pending or approved leave on Monday 5 October 2026.")
    mine = result("list_leave_requests", {"employee_id": "E1004", "requests": []}, status="pending")
    assert tool_answers.render([mine], "E1004") == "You have no pending leave requests."


def test_only_successful_read_results_are_rendered():
    ok = result("get_leave_balance", {"employee_id": "E1001", "balances": []})
    assert tool_answers.can_render([ok])
    assert not tool_answers.can_render([])
    assert not tool_answers.can_render([{**ok, "status": "error"}])
    assert not tool_answers.can_render([{**ok, "tool": "apply_leave"}])


TODAY = date(2026, 9, 15)


@pytest.mark.parametrize(
    "message, target",
    [
        ("How many annual leave days will I have by the end of December?", date(2026, 12, 31)),
        ("What will my balance be by the end of the year?", date(2026, 12, 31)),
        ("How much leave will I have by March?", date(2027, 3, 31)),  # March has passed: next year's
        ("Balance in November 2026?", date(2026, 11, 30)),
        ("How many days do I have?", None),
        ("How many days did I have in March 2026?", None),  # past: not a future balance
    ],
)
def test_target_date(message, target):
    assert find_target_date(message, TODAY) == target


@pytest.mark.parametrize(
    "message, expected",
    [
        ("How many annual leave days will I have by the end of December?", date(2026, 12, 31)),
        ("What will my balance be by the end of the year?", date(2026, 12, 31)),
        ("How many days do I have?", None),  # no future date: a current balance
        ("How many holidays will I have in December?", None),  # holidays, not a balance
        ("Is 31 December a holiday?", None),
    ],
)
def test_asks_future_balance(message, expected):
    assert asks_future_balance(message, TODAY) == expected


def test_calendar_note_counts_dates_not_rules():
    note = calendar_note(TODAY, date(2026, 12, 31), "2025-11-03")
    assert "4 (2026-09-30, 2026-10-31, 2026-11-30, 2026-12-31) month-end dates" in note
    assert "3 (2026-10-01, 2026-11-01, 2026-12-01) first days of a month" in note
    assert "0 years 10 months today, 1 year 1 month on 2026-12-31" in note
    assert "Service anniversary in this period: 2026-11-03." in note


def _fact(topic, attribute, value):
    from types import SimpleNamespace

    return SimpleNamespace(topic=topic, attribute=attribute, value=value)


@pytest.mark.parametrize(
    "facts, expected",
    [
        ([_fact("annual leave accrual", "credit date", "last calendar day of each month")], "end"),
        ([_fact("annual leave", "accrual credit day", "the first day of the month")], "start"),
        ([_fact("sick leave", "credit date", "1 January")], None),  # sick leave is credited yearly
        ([_fact("annual leave", "days per year", "18")], None),  # no credit-day fact
        (
            [_fact("annual leave", "credit date", "last day of the month"), _fact("annual leave", "credit date", "1st of the month")],
            None,  # disagreeing facts: let the documents decide
        ),
    ],
)
def test_credit_day_from_eligible_facts(facts, expected):
    assert credit_day(facts) == expected


def test_calendar_note_lists_only_the_matching_credit_dates():
    note = calendar_note(TODAY, date(2026, 12, 31), "2025-11-03", "end")
    assert "4 (2026-09-30, 2026-10-31, 2026-11-30, 2026-12-31) credit dates on the last day of the month" in note
    assert "first days of a month" not in note and "2026-10-01" not in note
    start = calendar_note(TODAY, date(2026, 12, 31), "2025-11-03", "start")
    assert "3 (2026-10-01, 2026-11-01, 2026-12-01) credit dates on the first day of the month" in start


def test_today_itself_is_not_counted():
    note = calendar_note(date(2026, 9, 30), date(2026, 10, 31), "2020-01-01")
    assert "1 (2026-10-31) month-end dates" in note and "1 (2026-10-01) first days" in note
