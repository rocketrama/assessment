"""POST /chat and GET /health contract (TASK.pdf §9). The graders' tests are automated,
so status codes and field names must match exactly."""

import threading

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import wait_ready

HEADERS = {"X-Employee-Id": "E1001"}
ROUTES = {"rag", "tool", "hybrid", "clarify", "no_answer", "refuse"}


def chat(client, body, headers=HEADERS):
    return client.post("/chat", json=body, headers=headers)


# --- /health (§9.2) ---------------------------------------------------------

def test_health_ok_when_ready(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_health_503_until_ingestion_finishes(settings):
    release = threading.Event()
    with TestClient(create_app(settings, ingest=lambda s: release.wait(5))) as c:
        assert c.get("/health").status_code == 503
        assert chat(c, {"session_id": "s1", "message": "hi"}).status_code == 503
        release.set()
        wait_ready(c)
        assert c.get("/health").status_code == 200


def test_health_stays_503_when_ingestion_fails(settings):
    def broken(_settings):
        raise RuntimeError("bad corpus")

    with TestClient(create_app(settings, ingest=broken)) as c:
        assert c.get("/health").status_code == 503


def test_missing_database_is_not_created(settings, tmp_path):
    settings.db_path = str(tmp_path / "missing.db")
    with TestClient(create_app(settings)) as c:
        assert c.get("/health").status_code == 503
    assert not (tmp_path / "missing.db").exists()


# --- 400: missing or invalid body (not FastAPI's default 422) ----------------

@pytest.mark.parametrize(
    "body",
    [
        None,
        {},
        {"session_id": "s1"},
        {"message": "hi"},
        {"session_id": "", "message": "hi"},
        {"session_id": "s1", "message": ""},
        {"session_id": 123, "message": "hi"},
        {"session_id": "s1", "message": ["hi"]},
    ],
)
def test_invalid_body_is_400(client, body):
    assert chat(client, body).status_code == 400


def test_malformed_json_is_400(client):
    resp = client.post("/chat", content=b"{not json", headers={**HEADERS, "Content-Type": "application/json"})
    assert resp.status_code == 400


# --- 401: missing or unknown employee (not 422) -----------------------------

@pytest.mark.parametrize("headers", [{}, {"X-Employee-Id": ""}, {"X-Employee-Id": "E9999"}])
def test_missing_or_unknown_employee_is_401(client, headers):
    assert chat(client, {"session_id": "s1", "message": "hi"}, headers=headers).status_code == 401


# --- 403: session owned by another employee ---------------------------------

def test_session_belongs_to_first_sender(client):
    assert chat(client, {"session_id": "s-own", "message": "hi"}).status_code == 200
    assert chat(client, {"session_id": "s-own", "message": "again"}).status_code == 200
    other = chat(client, {"session_id": "s-own", "message": "hi"}, headers={"X-Employee-Id": "E1002"})
    assert other.status_code == 403


# --- 200: response shape ----------------------------------------------------

def test_response_matches_contract(client):
    resp = chat(client, {"session_id": "s-shape", "message": "How many sick days do I get?"})
    assert resp.status_code == 200
    data = resp.json()
    assert set(data) == {"request_id", "answer", "route", "citations", "tool_calls", "pending_confirmation"}
    assert isinstance(data["request_id"], str) and data["request_id"]
    assert data["route"] in ROUTES
    assert isinstance(data["citations"], list)
    assert isinstance(data["tool_calls"], list)
    assert isinstance(data["pending_confirmation"], bool)
