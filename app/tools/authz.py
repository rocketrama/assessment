"""Every leave-data permission decision (TASK.pdf §3.1, §6.1, §7.3).

The requester always comes from the X-Employee-Id header via the DB, never from
LLM arguments. A target the requester may not see and a target that does not
exist get the same answer, so the refusal is no existence oracle.
"""

import sqlite3

from app.data.employees import Employee, get_employee
from app.data.leave import LeaveRequest

ACCESS_DENIED = "ACCESS_DENIED"
ACCESS_DENIED_MESSAGE = "I can't share information about that employee."

REQUEST_NOT_FOUND = "REQUEST_NOT_FOUND"
REQUEST_NOT_FOUND_MESSAGE = "I can't find a leave request with that ID among your requests."


def can_view_employee(conn: sqlite3.Connection, requester: Employee, target_id: str) -> bool:
    """Self always; a manager their direct reports; HR any one employee."""
    if target_id == requester.employee_id:
        return True
    target = get_employee(conn, target_id)
    if target is None:
        return False
    if requester.role == "hr":
        return True
    if requester.role == "manager":
        return target.manager_id == requester.employee_id
    return False


def can_view_team(requester: Employee) -> bool:
    """get_team_leave is for managers only; the query itself is scoped to their reports."""
    return requester.role == "manager"


def owns_request(requester: Employee, request: LeaveRequest | None) -> bool:
    """cancel_leave only on the requester's own request. A missing request and
    someone else's request get the same REQUEST_NOT_FOUND."""
    return request is not None and request.employee_id == requester.employee_id
