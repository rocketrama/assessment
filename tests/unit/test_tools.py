"""The five leave tools end to end against the synthetic DB (TASK.pdf §6)."""

import json
import threading
import uuid
from dataclasses import asdict
from datetime import date

import pytest

from app.data.db import connect, transaction
from app.data.employees import get_employee
from app.data.leave import get_balances, get_request, list_requests
from app.tools.authz import ACCESS_DENIED_MESSAGE, REQUEST_NOT_FOUND_MESSAGE
from app.tools.registry import (
    TOOLS,
    ToolContext,
    execute_apply_leave,
    execute_cancel_leave,
    get_leave_balance,
    get_team_leave,
    list_leave_requests,
    validate_apply_leave,
    validate_cancel_leave,
)
from tests.fixtures.fixture_db import EMPLOYEES
from tests.fixtures.holidays import TEST_HOLIDAYS, StaticHolidays

TODAY = date(2026, 9, 15)
NOV_2_6 = {"leave_type": "annual", "start_date": "2026-11-02", "end_date": "2026-11-06"}


def ctx_for(conn, employee_id, holidays=TEST_HOLIDAYS):
    return ToolContext(get_employee(conn, employee_id), conn, TODAY, holidays)


def annual(conn, employee_id="E1001"):
    return get_balances(conn, employee_id, "annual")[0].balance_days


# --- registry ---------------------------------------------------------------

def test_exact_tool_and_argument_names():
    assert {name: sorted(t.args_model.model_fields) for name, t in TOOLS.items()} == {
        "get_leave_balance": ["employee_id", "leave_type"],
        "list_leave_requests": ["employee_id", "status"],
        "get_team_leave": ["end_date", "start_date"],
        "apply_leave": ["end_date", "leave_type", "start_date"],
        "cancel_leave": ["request_id"],
    }
    assert {n for n, t in TOOLS.items() if t.writes} == {"apply_leave", "cancel_leave"}


@pytest.mark.parametrize(
    "args",
    [
        {"employee_id": 1001},  # no int → str coercion
        {"employee_id": "E1001", "role": "hr"},  # unknown key
        {},
    ],
)
def test_strict_arguments(db_conn, args):
    result = get_leave_balance(ctx_for(db_conn, "E1001"), args)
    assert (result.status, result.code) == ("error", "INVALID_ARGS")


# --- read tools -------------------------------------------------------------

def test_balance_self_all_and_by_type(db_conn):
    ctx = ctx_for(db_conn, "E1001")
    assert get_leave_balance(ctx, {"employee_id": "E1001"}).data["balances"] == [
        {"leave_type": "annual", "balance_days": 12.0},
        {"leave_type": "sick", "balance_days": 6.0},
    ]
    sick = get_leave_balance(ctx, {"employee_id": "E1001", "leave_type": "sick"})
    assert sick.data["balances"] == [{"leave_type": "sick", "balance_days": 6.0}]
    bad = get_leave_balance(ctx, {"employee_id": "E1001", "leave_type": "casual"})
    assert bad.code == "INVALID_LEAVE_TYPE"


@pytest.mark.parametrize("tool", [get_leave_balance, list_leave_requests])
def test_denied_and_missing_employee_get_identical_refusal(db_conn, tool):
    ctx = ctx_for(db_conn, "E1001")
    denied = tool(ctx, {"employee_id": "E1003"})
    missing = tool(ctx, {"employee_id": "E9999"})
    assert denied == missing
    assert (denied.status, denied.reason) == ("error", ACCESS_DENIED_MESSAGE)


def test_manager_sees_report_not_others(db_conn):
    ctx = ctx_for(db_conn, "M2001")
    assert get_leave_balance(ctx, {"employee_id": "E1001"}).status == "success"
    assert get_leave_balance(ctx, {"employee_id": "E1003"}).reason == ACCESS_DENIED_MESSAGE


def test_hr_sees_any_one_employee(db_conn):
    result = list_leave_requests(ctx_for(db_conn, "H9001"), {"employee_id": "E1003"})
    assert [r["request_id"] for r in result.data["requests"]] == ["REQ-FIX-0004"]


def test_list_requests_status_filter(db_conn):
    ctx = ctx_for(db_conn, "E1001")
    result = list_leave_requests(ctx, {"employee_id": "E1001", "status": "approved"})
    assert [r["request_id"] for r in result.data["requests"]] == ["REQ-FIX-0002", "REQ-FIX-0001"]


def test_team_leave_for_manager(db_conn):
    result = get_team_leave(ctx_for(db_conn, "M2001"), {"start_date": "2026-10-01", "end_date": "2026-10-31"})
    assert [r["request_id"] for r in result.data["requests"]] == ["REQ-FIX-0001", "REQ-FIX-0003"]


@pytest.mark.parametrize("requester", ["E1001", "H9001"])
def test_team_leave_refused_for_non_managers(db_conn, requester):
    result = get_team_leave(ctx_for(db_conn, requester), {"start_date": "2026-10-01", "end_date": "2026-10-31"})
    assert result.code == "NOT_A_MANAGER"


def test_team_leave_bad_range(db_conn):
    ctx = ctx_for(db_conn, "M2001")
    assert get_team_leave(ctx, {"start_date": "2026-10-31", "end_date": "2026-10-01"}).code == "END_BEFORE_START"
    assert get_team_leave(ctx, {"start_date": "next week", "end_date": "2026-10-01"}).code == "INVALID_DATE"


# --- apply_leave ------------------------------------------------------------

def test_validate_apply_is_a_dry_run(db_conn):
    result = validate_apply_leave(ctx_for(db_conn, "E1001"), NOV_2_6)
    assert result.status == "success"
    assert result.data["working_days"] == 4  # minus the fake IN holiday on 4 Nov
    assert (result.data["balance_days"], result.data["balance_after"]) == (12.0, 8.0)
    assert len(list_requests(db_conn, "E1001")) == 2
    assert annual(db_conn) == 12.0


def test_working_days_use_the_requesters_region(db_conn):
    args = {**NOV_2_6}
    assert validate_apply_leave(ctx_for(db_conn, "E1003"), args).data["working_days"] == 3  # DE
    assert validate_apply_leave(ctx_for(db_conn, "E1002"), args).data["working_days"] == 5  # US


def test_execute_apply_creates_pending_uuid_request_and_deducts(db_conn):
    with transaction(db_conn):
        result = execute_apply_leave(ctx_for(db_conn, "E1001"), NOV_2_6)
    assert result.status == "success"
    request_id = result.data["request_id"]
    assert uuid.UUID(request_id).version == 4
    stored = get_request(db_conn, request_id)
    assert (stored.status, stored.working_days, stored.employee_id) == ("pending", 4, "E1001")
    assert annual(db_conn) == 8.0 == result.data["balance_after"]


def test_execute_requires_a_transaction(db_conn):
    with pytest.raises(RuntimeError):
        execute_apply_leave(ctx_for(db_conn, "E1001"), NOV_2_6)
    with pytest.raises(RuntimeError):
        execute_cancel_leave(ctx_for(db_conn, "E1001"), {"request_id": "REQ-FIX-0001"})


@pytest.mark.parametrize(
    "args, code",
    [
        ({"leave_type": "annual", "start_date": "2026-10-06", "end_date": "2026-10-06"}, "OVERLAP"),
        ({"leave_type": "sick", "start_date": "2026-11-02", "end_date": "2026-11-13"}, "INSUFFICIENT_BALANCE"),
        ({"leave_type": "annual", "start_date": "2026-11-07", "end_date": "2026-11-08"}, "NO_WORKING_DAYS"),
        ({"leave_type": "annual", "start_date": "2026-11-04", "end_date": "2026-11-04"}, "NO_WORKING_DAYS"),
        ({"leave_type": "annual", "start_date": "2026-09-01", "end_date": "2026-09-02"}, "START_IN_PAST"),
        ({"leave_type": "annual", "start_date": "2026-12-28", "end_date": "2027-01-05"}, "OUTSIDE_LEAVE_YEAR"),
        ({"leave_type": "annual", "start_date": "2026-11-06", "end_date": "2026-11-02"}, "END_BEFORE_START"),
        ({"leave_type": "casual", "start_date": "2026-11-02", "end_date": "2026-11-02"}, "INVALID_LEAVE_TYPE"),
        ({"leave_type": "annual", "start_date": "2026-11-02"}, "INVALID_ARGS"),
        ({**NOV_2_6, "employee_id": "E1002"}, "INVALID_ARGS"),  # no applying for someone else
    ],
)
def test_apply_rejections_write_nothing(db_conn, args, code):
    with transaction(db_conn):
        result = execute_apply_leave(ctx_for(db_conn, "E1001"), args)
    assert (result.status, result.code) == ("error", code)
    assert result.reason
    assert len(list_requests(db_conn, "E1001")) == 2
    assert annual(db_conn) == 12.0


def test_execute_rechecks_rules_after_validate(db_conn):
    ctx = ctx_for(db_conn, "E1001")
    assert validate_apply_leave(ctx, NOV_2_6).status == "success"
    db_conn.execute("UPDATE leave_balances SET balance_days = 2 WHERE employee_id = 'E1001' AND leave_type = 'annual'")
    with transaction(db_conn):
        result = execute_apply_leave(ctx, NOV_2_6)
    assert result.code == "INSUFFICIENT_BALANCE"
    assert len(list_requests(db_conn, "E1001")) == 2


# --- cancel_leave -----------------------------------------------------------

def test_cancel_refunds_working_days(db_conn):
    ctx = ctx_for(db_conn, "E1001")
    assert validate_cancel_leave(ctx, {"request_id": "REQ-FIX-0001"}).status == "success"
    with transaction(db_conn):
        result = execute_cancel_leave(ctx, {"request_id": "REQ-FIX-0001"})
    assert result.status == "success"
    assert get_request(db_conn, "REQ-FIX-0001").status == "cancelled"
    assert annual(db_conn) == 15.0 == result.data["balance_after"]


def test_cancel_twice_refunds_once(db_conn):
    ctx = ctx_for(db_conn, "E1001")
    with transaction(db_conn):
        execute_cancel_leave(ctx, {"request_id": "REQ-FIX-0001"})
    with transaction(db_conn):
        second = execute_cancel_leave(ctx, {"request_id": "REQ-FIX-0001"})
    assert second.code == "NOT_CANCELLABLE_STATUS"
    assert annual(db_conn) == 15.0


def test_cancel_started_leave_rejected(db_conn):
    result = validate_cancel_leave(ctx_for(db_conn, "E1001"), {"request_id": "REQ-FIX-0002"})
    assert result.code == "ALREADY_STARTED"


@pytest.mark.parametrize("requester", ["E1001", "M2001", "H9001"])
def test_cancel_someone_elses_request_looks_like_not_found(db_conn, requester):
    ctx = ctx_for(db_conn, requester)
    others = validate_cancel_leave(ctx, {"request_id": "REQ-FIX-0003"})
    missing = validate_cancel_leave(ctx, {"request_id": "REQ-NOPE"})
    assert others == missing
    assert (others.status, others.reason) == ("error", REQUEST_NOT_FOUND_MESSAGE)


# --- privacy ----------------------------------------------------------------

def test_results_contain_no_personal_data(db_conn):
    results = [
        get_leave_balance(ctx_for(db_conn, "H9001"), {"employee_id": "E1001"}),
        list_leave_requests(ctx_for(db_conn, "M2001"), {"employee_id": "E1002"}),
        get_team_leave(ctx_for(db_conn, "M2001"), {"start_date": "2026-01-01", "end_date": "2026-12-31"}),
        validate_apply_leave(ctx_for(db_conn, "E1001"), NOV_2_6),
        validate_cancel_leave(ctx_for(db_conn, "E1001"), {"request_id": "REQ-FIX-0001"}),
    ]
    text = json.dumps([asdict(r) for r in results])
    for _id, name, email, phone, *_ in EMPLOYEES:
        assert name not in text and email not in text and phone not in text


# --- concurrency ------------------------------------------------------------

def _race(db_path, employee_id, calls):
    """Run each (tool, args) on its own connection, released together by a barrier."""
    barrier = threading.Barrier(len(calls))
    results = [None] * len(calls)

    def worker(i, tool, args):
        with connect(db_path) as conn:
            ctx = ToolContext(get_employee(conn, employee_id), conn, TODAY, StaticHolidays())
            barrier.wait()
            with transaction(conn):
                results[i] = tool(ctx, args)

    threads = [threading.Thread(target=worker, args=(i, t, a)) for i, (t, a) in enumerate(calls)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    return results


def test_concurrent_applies_cannot_overdraw(db_path):
    # Two non-overlapping 7-working-day requests against a balance of 12.
    a = {"leave_type": "annual", "start_date": "2026-11-02", "end_date": "2026-11-10"}
    b = {"leave_type": "annual", "start_date": "2026-11-16", "end_date": "2026-11-24"}
    results = _race(db_path, "E1001", [(execute_apply_leave, a), (execute_apply_leave, b)])
    assert sorted(r.status for r in results) == ["error", "success"]
    assert next(r for r in results if r.status == "error").code == "INSUFFICIENT_BALANCE"
    with connect(db_path) as conn:
        assert annual(conn) == 5.0


def test_concurrent_cancels_refund_once(db_path):
    args = {"request_id": "REQ-FIX-0001"}
    results = _race(db_path, "E1001", [(execute_cancel_leave, args), (execute_cancel_leave, args)])
    assert sorted(r.status for r in results) == ["error", "success"]
    with connect(db_path) as conn:
        assert annual(conn) == 15.0
