"""Leave balances and requests repository.

Data access only: permission checks live in tools/authz.py and rule checks in
tools/leave_rules.py. Write functions must run inside `db.transaction()` so the
apply/cancel flow can re-validate and write under one `BEGIN IMMEDIATE` lock.
"""

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

# Statuses that hold days and block overlapping requests.
ACTIVE_STATUSES = ("pending", "approved")

_REQUEST_FIELDS = (
    "request_id", "employee_id", "leave_type", "start_date",
    "end_date", "working_days", "status", "created_at",
)
_REQUEST_COLUMNS = ", ".join(_REQUEST_FIELDS)
_REQUEST_COLUMNS_R = ", ".join(f"r.{f}" for f in _REQUEST_FIELDS)


@dataclass(frozen=True)
class LeaveBalance:
    employee_id: str
    leave_type: str
    balance_days: float


@dataclass(frozen=True)
class LeaveRequest:
    request_id: str
    employee_id: str
    leave_type: str
    start_date: str
    end_date: str
    working_days: int
    status: str
    created_at: str


def _require_transaction(conn: sqlite3.Connection) -> None:
    if not conn.in_transaction:
        raise RuntimeError("leave writes must run inside db.transaction()")


def get_balances(
    conn: sqlite3.Connection, employee_id: str, leave_type: str | None = None
) -> list[LeaveBalance]:
    sql = "SELECT employee_id, leave_type, balance_days FROM leave_balances WHERE employee_id = ?"
    params: list[str] = [employee_id]
    if leave_type is not None:
        sql += " AND leave_type = ?"
        params.append(leave_type)
    rows = conn.execute(sql + " ORDER BY leave_type", params).fetchall()
    return [LeaveBalance(**dict(r)) for r in rows]


def get_request(conn: sqlite3.Connection, request_id: str) -> LeaveRequest | None:
    row = conn.execute(
        f"SELECT {_REQUEST_COLUMNS} FROM leave_requests WHERE request_id = ?", (request_id,)
    ).fetchone()
    return LeaveRequest(**dict(row)) if row else None


def list_requests(
    conn: sqlite3.Connection, employee_id: str, status: str | None = None
) -> list[LeaveRequest]:
    sql = f"SELECT {_REQUEST_COLUMNS} FROM leave_requests WHERE employee_id = ?"
    params: list[str] = [employee_id]
    if status is not None:
        sql += " AND status = ?"
        params.append(status)
    rows = conn.execute(sql + " ORDER BY start_date, request_id", params).fetchall()
    return [LeaveRequest(**dict(r)) for r in rows]


def overlapping_requests(
    conn: sqlite3.Connection, employee_id: str, start_date: str, end_date: str
) -> list[LeaveRequest]:
    """The employee's pending/approved requests that share at least one day with
    [start_date, end_date]. ISO dates compare correctly as strings."""
    rows = conn.execute(
        f"SELECT {_REQUEST_COLUMNS} FROM leave_requests "
        "WHERE employee_id = ? AND status IN (?, ?) AND start_date <= ? AND end_date >= ? "
        "ORDER BY start_date, request_id",
        (employee_id, *ACTIVE_STATUSES, end_date, start_date),
    ).fetchall()
    return [LeaveRequest(**dict(r)) for r in rows]


def team_requests(
    conn: sqlite3.Connection, manager_id: str, start_date: str, end_date: str
) -> list[LeaveRequest]:
    """Pending/approved requests of the manager's direct reports overlapping the range."""
    rows = conn.execute(
        f"SELECT {_REQUEST_COLUMNS_R} FROM leave_requests r JOIN employees e ON e.employee_id = r.employee_id "
        "WHERE e.manager_id = ? AND r.status IN (?, ?) AND r.start_date <= ? AND r.end_date >= ? "
        "ORDER BY r.start_date, r.employee_id, r.request_id",
        (manager_id, *ACTIVE_STATUSES, end_date, start_date),
    ).fetchall()
    return [LeaveRequest(**dict(r)) for r in rows]


def insert_request(
    conn: sqlite3.Connection,
    employee_id: str,
    leave_type: str,
    start_date: str,
    end_date: str,
    working_days: int,
) -> LeaveRequest:
    _require_transaction(conn)
    request = LeaveRequest(
        request_id=str(uuid.uuid4()),
        employee_id=employee_id,
        leave_type=leave_type,
        start_date=start_date,
        end_date=end_date,
        working_days=working_days,
        status="pending",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    conn.execute(
        f"INSERT INTO leave_requests ({_REQUEST_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            request.request_id,
            request.employee_id,
            request.leave_type,
            request.start_date,
            request.end_date,
            request.working_days,
            request.status,
            request.created_at,
        ),
    )
    return request


def set_status(conn: sqlite3.Connection, request_id: str, status: str) -> None:
    _require_transaction(conn)
    cur = conn.execute(
        "UPDATE leave_requests SET status = ? WHERE request_id = ?", (status, request_id)
    )
    if cur.rowcount != 1:
        raise LookupError(f"leave request not found: {request_id}")


def adjust_balance(
    conn: sqlite3.Connection, employee_id: str, leave_type: str, delta_days: float
) -> float:
    """Add `delta_days` (negative to deduct) and return the new balance."""
    _require_transaction(conn)
    cur = conn.execute(
        "UPDATE leave_balances SET balance_days = balance_days + ? "
        "WHERE employee_id = ? AND leave_type = ?",
        (delta_days, employee_id, leave_type),
    )
    if cur.rowcount != 1:
        raise LookupError(f"no {leave_type} balance for {employee_id}")
    return conn.execute(
        "SELECT balance_days FROM leave_balances WHERE employee_id = ? AND leave_type = ?",
        (employee_id, leave_type),
    ).fetchone()["balance_days"]
