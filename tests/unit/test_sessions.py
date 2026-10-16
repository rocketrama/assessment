"""Session history and the one-proposal-per-session store (ARCHITECTURE §10)."""

import pytest

from app.data.db import transaction
from app.data.sessions import (
    add_message,
    claim_session,
    delete_proposal,
    get_proposal,
    recent_messages,
    save_proposal,
    take_proposal,
)

ARGS = {"leave_type": "annual", "start_date": "2026-11-02", "end_date": "2026-11-03"}


@pytest.fixture
def session(db_conn):
    claim_session(db_conn, "s1", "E1001")
    return "s1"


def test_recent_messages_oldest_first_and_limited(db_conn, session):
    for i in range(8):
        add_message(db_conn, session, "user" if i % 2 == 0 else "assistant", f"m{i}")
    got = recent_messages(db_conn, session, limit=3)
    assert [m.content for m in got] == ["m5", "m6", "m7"]
    assert [m.role for m in got] == ["assistant", "user", "assistant"]


def test_messages_are_per_session(db_conn, session):
    claim_session(db_conn, "s2", "E1002")
    add_message(db_conn, session, "user", "mine")
    add_message(db_conn, "s2", "user", "theirs")
    assert [m.content for m in recent_messages(db_conn, session)] == ["mine"]


def test_unknown_role_rejected(db_conn, session):
    with pytest.raises(ValueError):
        add_message(db_conn, session, "system", "x")


def test_save_and_get_proposal_round_trips_args(db_conn, session):
    saved = save_proposal(db_conn, session, "E1001", "apply_leave", ARGS, "Annual leave 2–3 Nov")
    got = get_proposal(db_conn, session)
    assert got == saved
    assert got.args == ARGS


def test_new_proposal_replaces_old_one(db_conn, session):
    save_proposal(db_conn, session, "E1001", "apply_leave", ARGS, "first")
    save_proposal(db_conn, session, "E1001", "cancel_leave", {"request_id": "REQ-FIX-0001"}, "second")
    got = get_proposal(db_conn, session)
    assert (got.tool_name, got.summary) == ("cancel_leave", "second")
    assert db_conn.execute("SELECT COUNT(*) FROM proposals").fetchone()[0] == 1


def test_take_proposal_succeeds_only_once(db_conn, session):
    save_proposal(db_conn, session, "E1001", "apply_leave", ARGS, "s")
    with transaction(db_conn):
        first = take_proposal(db_conn, session)
    with transaction(db_conn):
        second = take_proposal(db_conn, session)
    assert first.args == ARGS
    assert second is None


def test_take_proposal_is_restored_when_the_write_fails(db_conn, session):
    save_proposal(db_conn, session, "E1001", "apply_leave", ARGS, "s")
    with pytest.raises(RuntimeError):
        with transaction(db_conn):
            take_proposal(db_conn, session)
            raise RuntimeError("leave write failed")
    assert get_proposal(db_conn, session) is not None


def test_take_proposal_requires_a_transaction(db_conn, session):
    with pytest.raises(RuntimeError):
        take_proposal(db_conn, session)


def test_delete_proposal(db_conn, session):
    save_proposal(db_conn, session, "E1001", "apply_leave", ARGS, "s")
    delete_proposal(db_conn, session)
    assert get_proposal(db_conn, session) is None
