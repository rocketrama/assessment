"""Employee repository. Personal data is minimised at the source: `phone` is never
selected, and names/emails are only exposed through requester-aware methods (step 2)."""

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class Employee:
    """The requester's identity, role and region — always from the DB (§7.1). No PII."""

    employee_id: str
    region: str
    role: str
    level: str
    manager_id: str | None
    join_date: str


def get_employee(conn: sqlite3.Connection, employee_id: str) -> Employee | None:
    row = conn.execute(
        "SELECT employee_id, region, role, level, manager_id, join_date "
        "FROM employees WHERE employee_id = ?",
        (employee_id,),
    ).fetchone()
    return Employee(**dict(row)) if row else None
