"""Synthetic veltrix.db: the exact schema from TASK.pdf §2.3 with fake people.

All names, emails and phone numbers are invented. Used by the test suite and by
`scripts/make_dev_db.py` for local runs, so the real database is never needed
during development.
"""

import sqlite3

SCHEMA = """
CREATE TABLE employees (
  employee_id TEXT PRIMARY KEY,
  full_name   TEXT NOT NULL,
  email       TEXT NOT NULL,
  phone       TEXT NOT NULL,
  region      TEXT NOT NULL,
  join_date   TEXT NOT NULL,
  level       TEXT NOT NULL,
  role        TEXT NOT NULL,
  manager_id  TEXT REFERENCES employees(employee_id)
);
CREATE TABLE leave_balances (
  employee_id  TEXT NOT NULL REFERENCES employees(employee_id),
  leave_type   TEXT NOT NULL,
  balance_days REAL NOT NULL,
  PRIMARY KEY (employee_id, leave_type)
);
CREATE TABLE leave_requests (
  request_id   TEXT PRIMARY KEY,
  employee_id  TEXT NOT NULL REFERENCES employees(employee_id),
  leave_type   TEXT NOT NULL,
  start_date   TEXT NOT NULL,
  end_date     TEXT NOT NULL,
  working_days INTEGER NOT NULL,
  status       TEXT NOT NULL,
  created_at   TEXT NOT NULL
);
"""

# (id, name, email, phone, region, join_date, level, role, manager_id)
EMPLOYEES = [
    ("H9001", "Test Hr One", "hr.one@veltrix.example", "+00-000-0001", "IN", "2020-01-06", "L6", "hr", None),
    ("M2001", "Test Manager In", "m.in@veltrix.example", "+00-000-0002", "IN", "2021-03-01", "L5", "manager", None),
    ("M2002", "Test Manager De", "m.de@veltrix.example", "+00-000-0003", "DE", "2021-05-03", "L5", "manager", None),
    ("E1001", "Test Employee In", "e.in@veltrix.example", "+00-000-0004", "IN", "2023-07-10", "L2", "employee", "M2001"),
    ("E1002", "Test Employee Us", "e.us@veltrix.example", "+00-000-0005", "US", "2024-02-12", "L3", "employee", "M2001"),
    ("E1003", "Test Employee De", "e.de@veltrix.example", "+00-000-0006", "DE", "2022-09-01", "L4", "employee", "M2002"),
    ("E1004", "Test Employee In Two", "e.in2@veltrix.example", "+00-000-0007", "IN", "2025-11-03", "L1", "employee", "M2002"),
]

# (employee_id, leave_type, balance_days)
BALANCES = [
    (emp[0], leave_type, days)
    for emp in EMPLOYEES
    for leave_type, days in (("annual", 12.0), ("sick", 6.0))
]

# (request_id, employee_id, leave_type, start, end, working_days, status, created_at)
REQUESTS = [
    ("REQ-FIX-0001", "E1001", "annual", "2026-10-05", "2026-10-07", 3, "approved", "2026-08-01T09:00:00Z"),
    ("REQ-FIX-0002", "E1001", "sick", "2026-08-03", "2026-08-03", 1, "approved", "2026-08-03T08:00:00Z"),
    ("REQ-FIX-0003", "E1002", "annual", "2026-10-06", "2026-10-09", 4, "pending", "2026-09-01T10:00:00Z"),
    ("REQ-FIX-0004", "E1003", "annual", "2026-12-21", "2026-12-23", 3, "cancelled", "2026-09-02T10:00:00Z"),
]


def build_fixture_db(path: str) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO employees VALUES (?,?,?,?,?,?,?,?,?)", EMPLOYEES)
        conn.executemany("INSERT INTO leave_balances VALUES (?,?,?)", BALANCES)
        conn.executemany("INSERT INTO leave_requests VALUES (?,?,?,?,?,?,?,?)", REQUESTS)
        conn.commit()
    finally:
        conn.close()
