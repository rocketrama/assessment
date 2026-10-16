"""Leave-data permissions (TASK.pdf §3.1, §7.3): each role × self / report / other / missing."""

import pytest

from app.data.employees import get_employee
from app.data.leave import get_request
from app.tools.authz import can_view_employee, can_view_team, owns_request

# Fixture org: M2001 manages E1001, E1002; M2002 manages E1003, E1004; H9001 is HR.


@pytest.mark.parametrize(
    "requester, target, allowed",
    [
        # employee
        ("E1001", "E1001", True),
        ("E1001", "E1002", False),  # colleague
        ("E1001", "M2001", False),  # own manager
        ("E1001", "E9999", False),  # does not exist
        # manager
        ("M2001", "M2001", True),
        ("M2001", "E1001", True),  # direct report
        ("M2001", "E1003", False),  # another manager's report
        ("M2001", "M2002", False),  # another manager
        ("M2001", "H9001", False),
        ("M2001", "E9999", False),
        # hr
        ("H9001", "H9001", True),
        ("H9001", "E1003", True),
        ("H9001", "M2001", True),
        ("H9001", "E9999", False),
    ],
)
def test_can_view_employee(db_conn, requester, target, allowed):
    assert can_view_employee(db_conn, get_employee(db_conn, requester), target) is allowed


@pytest.mark.parametrize("requester, allowed", [("M2001", True), ("E1001", False), ("H9001", False)])
def test_can_view_team_is_managers_only(db_conn, requester, allowed):
    assert can_view_team(get_employee(db_conn, requester)) is allowed


def test_owns_request(db_conn):
    me = get_employee(db_conn, "E1001")
    assert owns_request(me, get_request(db_conn, "REQ-FIX-0001"))
    assert not owns_request(me, get_request(db_conn, "REQ-FIX-0003"))  # E1002's
    assert not owns_request(me, None)
    # A manager or HR may not cancel someone else's leave either.
    assert not owns_request(get_employee(db_conn, "M2001"), get_request(db_conn, "REQ-FIX-0001"))
    assert not owns_request(get_employee(db_conn, "H9001"), get_request(db_conn, "REQ-FIX-0001"))
