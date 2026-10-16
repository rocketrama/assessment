"""Employee repository. Personal data is minimised at the source: `phone` is never
selected, emails leave this module masked, and names are only returned by
`get_contact` (callers decide visibility in tools/authz.py) and `known_names`
(for log redaction)."""

import sqlite3
from dataclasses import dataclass

from app.privacy.masking import mask_email

_EMPLOYEE_COLUMNS = "employee_id, region, role, level, manager_id, join_date"


@dataclass(frozen=True)
class Employee:
    """The requester's identity, role and region — always from the DB (§7.1). No PII."""

    employee_id: str
    region: str
    role: str
    level: str
    manager_id: str | None
    join_date: str


@dataclass(frozen=True)
class Contact:
    """Name and masked email of one employee; only for people the requester may see."""

    employee_id: str
    full_name: str
    email_masked: str


def get_employee(conn: sqlite3.Connection, employee_id: str) -> Employee | None:
    row = conn.execute(
        f"SELECT {_EMPLOYEE_COLUMNS} FROM employees WHERE employee_id = ?",
        (employee_id,),
    ).fetchone()
    return Employee(**dict(row)) if row else None


def get_direct_reports(conn: sqlite3.Connection, manager_id: str) -> list[Employee]:
    rows = conn.execute(
        f"SELECT {_EMPLOYEE_COLUMNS} FROM employees WHERE manager_id = ? ORDER BY employee_id",
        (manager_id,),
    ).fetchall()
    return [Employee(**dict(r)) for r in rows]


def get_contact(conn: sqlite3.Connection, employee_id: str) -> Contact | None:
    row = conn.execute(
        "SELECT employee_id, full_name, email FROM employees WHERE employee_id = ?",
        (employee_id,),
    ).fetchone()
    if row is None:
        return None
    return Contact(row["employee_id"], row["full_name"], mask_email(row["email"]))


def known_names(conn: sqlite3.Connection) -> list[str]:
    """Every full name, for the log redaction filter only."""
    return [r["full_name"] for r in conn.execute("SELECT full_name FROM employees")]
