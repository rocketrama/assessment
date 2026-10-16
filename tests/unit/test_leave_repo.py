"""Employee and leave repositories against the synthetic DB (tests/fixtures/fixture_db.py)."""

import pytest

from app.data.db import transaction
from app.data.employees import get_contact, get_direct_reports, known_names
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

# --- employees --------------------------------------------------------------


def test_direct_reports(db_conn):
    assert [e.employee_id for e in get_direct_reports(db_conn, "M2001")] == ["E1001", "E1002"]
    assert get_direct_reports(db_conn, "E1001") == []


def test_contact_has_masked_email_and_no_phone(db_conn):
    contact = get_contact(db_conn, "E1001")
    assert contact.full_name == "Test Employee In"
    assert contact.email_masked == "e***@veltrix.example"
    assert not hasattr(contact, "phone")
    assert get_contact(db_conn, "NOPE") is None


def test_known_names(db_conn):
    assert "Test Hr One" in known_names(db_conn)


# --- reads ------------------------------------------------------------------


def test_balances_all_and_by_type(db_conn):
    assert [(b.leave_type, b.balance_days) for b in get_balances(db_conn, "E1001")] == [
        ("annual", 12.0),
        ("sick", 6.0),
    ]
    assert [b.leave_type for b in get_balances(db_conn, "E1001", "sick")] == ["sick"]
    assert get_balances(db_conn, "NOPE") == []


def test_list_requests_with_status_filter(db_conn):
    assert [r.request_id for r in list_requests(db_conn, "E1001")] == ["REQ-FIX-0002", "REQ-FIX-0001"]
    assert [r.request_id for r in list_requests(db_conn, "E1001", "approved")] == [
        "REQ-FIX-0002",
        "REQ-FIX-0001",
    ]
    assert list_requests(db_conn, "E1001", "pending") == []


def test_get_request(db_conn):
    assert get_request(db_conn, "REQ-FIX-0003").employee_id == "E1002"
    assert get_request(db_conn, "NOPE") is None


@pytest.mark.parametrize(
    "start, end, expected",
    [
        ("2026-10-07", "2026-10-09", ["REQ-FIX-0001"]),  # touches the last day
        ("2026-10-01", "2026-10-05", ["REQ-FIX-0001"]),  # touches the first day
        ("2026-10-08", "2026-10-09", []),  # starts the day after
        ("2026-08-03", "2026-08-03", ["REQ-FIX-0002"]),
    ],
)
def test_overlapping_requests_edges(db_conn, start, end, expected):
    assert [r.request_id for r in overlapping_requests(db_conn, "E1001", start, end)] == expected


def test_overlap_ignores_cancelled(db_conn):
    assert overlapping_requests(db_conn, "E1003", "2026-12-21", "2026-12-23") == []


def test_team_requests_only_direct_reports_active_and_overlapping(db_conn):
    got = team_requests(db_conn, "M2001", "2026-10-01", "2026-10-31")
    assert [r.request_id for r in got] == ["REQ-FIX-0001", "REQ-FIX-0003"]
    # M2002's report E1003 only has a cancelled request
    assert team_requests(db_conn, "M2002", "2026-12-01", "2026-12-31") == []


# --- writes -----------------------------------------------------------------


def test_writes_require_a_transaction(db_conn):
    with pytest.raises(RuntimeError):
        insert_request(db_conn, "E1001", "annual", "2026-11-02", "2026-11-03", 2)
    with pytest.raises(RuntimeError):
        adjust_balance(db_conn, "E1001", "annual", -1)
    with pytest.raises(RuntimeError):
        set_status(db_conn, "REQ-FIX-0001", "cancelled")


def test_apply_and_cancel_round_trip(db_conn):
    with transaction(db_conn):
        req = insert_request(db_conn, "E1001", "annual", "2026-11-02", "2026-11-03", 2)
        assert adjust_balance(db_conn, "E1001", "annual", -2) == 10.0
    stored = get_request(db_conn, req.request_id)
    assert stored == req
    assert stored.status == "pending"

    with transaction(db_conn):
        set_status(db_conn, req.request_id, "cancelled")
        assert adjust_balance(db_conn, "E1001", "annual", 2) == 12.0
    assert get_request(db_conn, req.request_id).status == "cancelled"


def test_failed_transaction_rolls_back_everything(db_conn):
    with pytest.raises(LookupError):
        with transaction(db_conn):
            insert_request(db_conn, "E1001", "annual", "2026-11-02", "2026-11-03", 2)
            adjust_balance(db_conn, "E1001", "annual", -2)
            adjust_balance(db_conn, "E1001", "parental", -1)  # no such balance row
    assert len(list_requests(db_conn, "E1001")) == 2
    assert get_balances(db_conn, "E1001", "annual")[0].balance_days == 12.0


def test_set_status_unknown_request(db_conn):
    with pytest.raises(LookupError):
        with transaction(db_conn):
            set_status(db_conn, "NOPE", "cancelled")
