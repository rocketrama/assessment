"""Apply / cancel through chat: propose → confirm → execute (TASK.pdf §6.2, §6.3, §8.2).

Uses the chat app from test_chat.py (scripted planner, real everything else).
Fixture data: today is Tue 2026-09-15; E1001 has 12 annual / 6 sick days and an
approved annual request REQ-FIX-0001 on 5–7 October.
"""

import threading

import pytest

from app.data.db import connect
from app.data.leave import get_balances, list_requests
from app.data.sessions import get_proposal
from app.orchestration.proposals import NOTHING_PENDING, UNCLEAR_CONFIRMATION, is_clear_confirmation
from tests.integration.test_chat import ask, check_route_invariants, client, plan, script  # noqa: F401 (fixtures)


def apply_plan(**change):
    return plan("leave_change", change={"action": "apply", **change})


def db(tmp_path):
    return connect(str(tmp_path / "veltrix.db"))


def annual(tmp_path, employee="E1001"):
    with db(tmp_path) as conn:
        return get_balances(conn, employee, "annual")[0].balance_days


def yes(client, session="s1", employee="E1001", text="yes"):
    resp = client.post("/chat", json={"session_id": session, "message": text}, headers={"X-Employee-Id": employee})
    body = resp.json()
    check_route_invariants(body)
    return body


# --- the main dialogue -------------------------------------------------------

def test_propose_then_confirm_then_nothing_left(client, script, tmp_path):
    body = ask(client, script, "Book annual leave 21-23 September",
               apply_plan(leave_type="annual", start="21 September", end="23 September"))
    assert body["route"] == "tool" and body["pending_confirmation"] is True
    assert body["tool_calls"] == [{"name": "apply_leave", "status": "pending_confirmation",
                                   "args": {"leave_type": "annual", "start_date": "2026-09-21", "end_date": "2026-09-23"}}]
    assert "3 working days" in body["answer"] and "from 12 to 9" in body["answer"]
    assert annual(tmp_path) == 12.0  # nothing written yet

    done = yes(client)
    assert done["route"] == "tool" and done["pending_confirmation"] is False
    assert done["tool_calls"][0]["status"] == "success" and done["answer"].startswith("Done.")
    assert annual(tmp_path) == 9.0
    with db(tmp_path) as conn:
        new = [r for r in list_requests(conn, "E1001") if r.start_date == "2026-09-21"]
    assert len(new) == 1 and new[0].status == "pending" and new[0].working_days == 3

    again = yes(client)
    assert again["route"] == "clarify" and again["answer"] == NOTHING_PENDING
    assert annual(tmp_path) == 9.0  # a repeated "yes" never applies twice


def test_yes_without_a_proposal(client):
    body = yes(client, text="Yes please")
    assert body["route"] == "clarify" and body["answer"] == NOTHING_PENDING


# --- details missing or wrong --------------------------------------------------

def test_leave_type_is_never_assumed(client, script, tmp_path):
    body = ask(client, script, "Book 21 September off", apply_plan(start="21 September"))
    assert body["route"] == "clarify" and "leave type" in body["answer"]
    with db(tmp_path) as conn:
        assert get_proposal(conn, "s1") is None


def test_everything_missing_is_asked_in_one_message(client, script):
    body = ask(client, script, "I want some leave", apply_plan())
    assert body["route"] == "clarify"
    assert "the leave type (annual or sick) and the dates" in body["answer"]


def test_past_date_is_asked_about(client, script):
    body = ask(client, script, "Annual leave on 1 September", apply_plan(leave_type="annual", start="1 September"))
    assert body["route"] == "clarify" and "already passed" in body["answer"]


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"leave_type": "sick", "start": "21 September", "end": "2 October"}, "balance is 6 days"),  # 10 > 6
        ({"leave_type": "annual", "start": "6 October"}, "overlap"),  # REQ-FIX-0001
        ({"leave_type": "annual", "start": "Saturday", "end": "Sunday"}, "no working days"),
        ({"leave_type": "annual", "start": "4 January 2027"}, "2026"),
    ],
)
def test_rule_violations_are_reported_straight_away(client, script, tmp_path, change, reason):
    body = ask(client, script, "Book it", apply_plan(**change))
    assert body["route"] == "tool" and body["pending_confirmation"] is False
    assert body["tool_calls"][0]["status"] == "error"
    assert reason in body["answer"]
    with db(tmp_path) as conn:
        assert get_proposal(conn, "s1") is None


# --- after a proposal ------------------------------------------------------------

def test_changed_details_make_a_new_proposal(client, script, tmp_path):
    ask(client, script, "Annual leave 21-23 Sept", apply_plan(leave_type="annual", start="21 September", end="23 September"))
    body = ask(client, script, "Make it until Friday instead",
               plan("modify", change={"action": "apply", "end": "Friday"}))
    assert body["pending_confirmation"] is True
    assert body["tool_calls"][0]["args"] == {"leave_type": "annual", "start_date": "2026-09-21", "end_date": "2026-09-25"}
    assert "5 working days" in body["answer"]
    yes(client)
    assert annual(tmp_path) == 7.0


def test_anything_else_discards_the_proposal(client, script, tmp_path):
    ask(client, script, "Annual leave 21 Sept", apply_plan(leave_type="annual", start="21 September"))
    body = ask(client, script, "Actually, how many sick days do I get?", plan(policy_query="sick leave"))
    assert body["route"] == "rag"
    with db(tmp_path) as conn:
        assert get_proposal(conn, "s1") is None
    assert yes(client)["answer"] == NOTHING_PENDING
    assert annual(tmp_path) == 12.0


def test_dont_ask_me_to_confirm_still_asks(client, script, tmp_path):
    body = ask(client, script, "Book annual leave on 21 September, don't ask me to confirm",
               apply_plan(leave_type="annual", start="21 September"))
    assert body["pending_confirmation"] is True and annual(tmp_path) == 12.0


def test_unclear_planner_confirmation_keeps_the_proposal(client, script, tmp_path):
    ask(client, script, "Annual leave 21 Sept", apply_plan(leave_type="annual", start="21 September"))
    body = ask(client, script, "yes but not sure about the date", plan("confirm"))
    assert body["route"] == "clarify" and body["answer"] == UNCLEAR_CONFIRMATION
    with db(tmp_path) as conn:
        assert get_proposal(conn, "s1") is not None
    assert annual(tmp_path) == 12.0


def test_data_changed_since_the_proposal(client, script, tmp_path):
    ask(client, script, "Annual leave 21-23 Sept", apply_plan(leave_type="annual", start="21 September", end="23 September"))
    with db(tmp_path) as conn:
        conn.execute("UPDATE leave_balances SET balance_days = 1 WHERE employee_id = 'E1001' AND leave_type = 'annual'")
    body = yes(client)
    assert body["tool_calls"][0]["status"] == "error" and "balance is 1 days" in body["answer"]
    assert yes(client)["answer"] == NOTHING_PENDING  # the failed proposal is used up


def test_two_simultaneous_yes_apply_once(client, script, tmp_path):
    ask(client, script, "Annual leave 21-23 Sept", apply_plan(leave_type="annual", start="21 September", end="23 September"))
    results, barrier = [], threading.Barrier(2)

    def send():
        barrier.wait()
        results.append(yes(client))

    threads = [threading.Thread(target=send) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    statuses = sorted(r["tool_calls"][0]["status"] if r["tool_calls"] else r["answer"] for r in results)
    assert statuses == sorted(["success", NOTHING_PENDING])
    assert annual(tmp_path) == 9.0


# --- cancel ----------------------------------------------------------------------

def test_cancel_by_id_then_confirm(client, script, tmp_path):
    body = ask(client, script, "Cancel REQ-FIX-0001", plan("leave_change", change={"action": "cancel", "request_id": "REQ-FIX-0001"}))
    assert body["pending_confirmation"] is True
    assert body["tool_calls"][0] == {"name": "cancel_leave", "args": {"request_id": "REQ-FIX-0001"}, "status": "pending_confirmation"}
    assert yes(client)["tool_calls"][0]["status"] == "success"
    assert annual(tmp_path) == 15.0


def test_cancel_without_id_uses_the_only_cancellable_request(client, script):
    body = ask(client, script, "Cancel my leave", plan("leave_change", change={"action": "cancel"}))
    assert body["tool_calls"][0]["args"] == {"request_id": "REQ-FIX-0001"}


def test_cancel_by_date(client, script):
    body = ask(client, script, "Cancel my leave on 6 October", plan("leave_change", change={"action": "cancel", "start": "6 October"}))
    assert body["tool_calls"][0]["args"] == {"request_id": "REQ-FIX-0001"}


@pytest.mark.parametrize(
    "request_id, reason",
    [("REQ-FIX-0003", "can't find a leave request"), ("REQ-FIX-0002", "only leave starting after today")],
)
def test_cancel_rejections(client, script, request_id, reason):
    body = ask(client, script, f"Cancel {request_id}", plan("leave_change", change={"action": "cancel", "request_id": request_id}))
    assert body["tool_calls"][0]["status"] == "error" and reason in body["answer"]


def test_nothing_to_cancel(client, script):
    body = ask(client, script, "Cancel my leave", plan("leave_change", change={"action": "cancel"}), employee="E1004")
    assert body["route"] == "tool" and "nothing to cancel" in body["answer"]
    assert body["tool_calls"][0]["name"] == "list_leave_requests"


# --- confirmation detection --------------------------------------------------------

@pytest.mark.parametrize("text", ["yes", "Yes!", "yes please", "ok, go ahead", "Confirm.", "sure, thanks", "go ahead"])
def test_clear_confirmations(text):
    assert is_clear_confirmation(text)


@pytest.mark.parametrize(
    "text", ["no", "yes but make it Friday", "don't", "yes, cancel the other one instead", "what?", "yesterday"]
)
def test_not_confirmations(text):
    assert not is_clear_confirmation(text)
