"""Session state (§7.1, DESIGN §5): ownership, the last few turns, and at most one
pending proposal per session.

A session belongs to the employee who sent its first message. Proposals hold the
server-side args of a write action; on confirmation only these stored args run.
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.data.db import transaction

MESSAGE_ROLES = ("user", "assistant")


class SessionOwnershipError(Exception):
    """The session_id belongs to another employee → HTTP 403."""


@dataclass(frozen=True)
class Message:
    role: str
    content: str
    created_at: str


@dataclass(frozen=True)
class Proposal:
    session_id: str
    employee_id: str
    tool_name: str
    args: dict[str, Any]
    summary: str
    created_at: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def claim_session(conn: sqlite3.Connection, session_id: str, employee_id: str) -> None:
    """Create the session for this employee, or verify they already own it.

    Insert-then-read inside one write transaction, so two first messages racing
    for the same session_id can't both win.
    """
    with transaction(conn):
        conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, employee_id, created_at) VALUES (?, ?, ?)",
            (session_id, employee_id, _now()),
        )
        owner = conn.execute(
            "SELECT employee_id FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()["employee_id"]
    if owner != employee_id:
        raise SessionOwnershipError(session_id)


# --- history ----------------------------------------------------------------

def add_message(conn: sqlite3.Connection, session_id: str, role: str, content: str) -> None:
    if role not in MESSAGE_ROLES:
        raise ValueError(f"unknown message role: {role}")
    conn.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (session_id, role, content, _now()),
    )


def recent_messages(conn: sqlite3.Connection, session_id: str, limit: int = 6) -> list[Message]:
    """The last `limit` messages, oldest first. rowid breaks same-timestamp ties."""
    rows = conn.execute(
        "SELECT role, content, created_at FROM messages WHERE session_id = ? "
        "ORDER BY created_at DESC, rowid DESC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    return [Message(**dict(r)) for r in reversed(rows)]


# --- proposals --------------------------------------------------------------

def _to_proposal(row: sqlite3.Row) -> Proposal:
    data = dict(row)
    data["args"] = json.loads(data.pop("args_json"))
    return Proposal(**data)


def save_proposal(
    conn: sqlite3.Connection,
    session_id: str,
    employee_id: str,
    tool_name: str,
    args: dict[str, Any],
    summary: str,
) -> Proposal:
    """Store the pending write, replacing any earlier one (a modified request
    supersedes the old proposal; the primary key allows only one)."""
    proposal = Proposal(session_id, employee_id, tool_name, dict(args), summary, _now())
    conn.execute(
        "INSERT OR REPLACE INTO proposals "
        "(session_id, employee_id, tool_name, args_json, summary, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            proposal.session_id,
            proposal.employee_id,
            proposal.tool_name,
            json.dumps(proposal.args, sort_keys=True),
            proposal.summary,
            proposal.created_at,
        ),
    )
    return proposal


def get_proposal(conn: sqlite3.Connection, session_id: str) -> Proposal | None:
    row = conn.execute(
        "SELECT session_id, employee_id, tool_name, args_json, summary, created_at "
        "FROM proposals WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return _to_proposal(row) if row else None


def take_proposal(conn: sqlite3.Connection, session_id: str) -> Proposal | None:
    """Claim the proposal for execution: read and delete it.

    Must run inside the same `db.transaction()` as the leave write, so a duplicate
    "yes" finds nothing and the write can never run twice (ARCHITECTURE §10).
    """
    if not conn.in_transaction:
        raise RuntimeError("take_proposal must run inside db.transaction()")
    proposal = get_proposal(conn, session_id)
    if proposal is not None:
        conn.execute("DELETE FROM proposals WHERE session_id = ?", (session_id,))
    return proposal


def delete_proposal(conn: sqlite3.Connection, session_id: str) -> None:
    """Discard without executing (the user moved on to something else)."""
    conn.execute("DELETE FROM proposals WHERE session_id = ?", (session_id,))
