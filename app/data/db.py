"""SQLite connections, transactions and the tables this service adds to veltrix.db.

Existing tables (employees, leave_balances, leave_requests) are never changed
(TASK.pdf §2.3). Sessions, messages and proposals live in veltrix.db because a
proposal is claimed in the same transaction as the leave write it executes.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

APP_TABLES = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    employee_id TEXT NOT NULL REFERENCES employees(employee_id),
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);
CREATE TABLE IF NOT EXISTS proposals (
    session_id  TEXT PRIMARY KEY REFERENCES sessions(session_id),
    employee_id TEXT NOT NULL,
    tool_name   TEXT NOT NULL,
    args_json   TEXT NOT NULL,
    summary     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""


@contextmanager
def connect(db_path: str) -> Iterator[sqlite3.Connection]:
    """One connection per unit of work; never shared across threads.

    `isolation_level=None` stops Python's sqlite3 module from opening implicit
    transactions, so `transaction()` controls BEGIN/COMMIT explicitly.
    """
    conn = sqlite3.connect(db_path, isolation_level=None, timeout=5.0)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
    finally:
        conn.close()


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """`BEGIN IMMEDIATE`: take the write lock before reading, so check-then-act can't race."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def init_app_tables(db_path: str) -> None:
    with connect(db_path) as conn:
        conn.executescript(APP_TABLES)
