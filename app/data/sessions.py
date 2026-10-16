"""Session ownership: a session belongs to the employee who sent its first message (§7.1)."""

import sqlite3
from datetime import datetime, timezone

from app.data.db import transaction


class SessionOwnershipError(Exception):
    """The session_id belongs to another employee → HTTP 403."""


def claim_session(conn: sqlite3.Connection, session_id: str, employee_id: str) -> None:
    """Create the session for this employee, or verify they already own it.

    Insert-then-read inside one write transaction, so two first messages racing
    for the same session_id can't both win.
    """
    now = datetime.now(timezone.utc).isoformat()
    with transaction(conn):
        conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, employee_id, created_at) VALUES (?, ?, ?)",
            (session_id, employee_id, now),
        )
        owner = conn.execute(
            "SELECT employee_id FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()["employee_id"]
    if owner != employee_id:
        raise SessionOwnershipError(session_id)
