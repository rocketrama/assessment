"""Structured logs: JSON with request fields, and never the user's message text."""

import json
import logging

from app.logging_setup import JsonFormatter


class _Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(JsonFormatter().format(record))


def test_json_formatter_puts_extra_fields_at_top_level():
    record = logging.makeLogRecord(
        {"name": "veltrix", "levelname": "INFO", "msg": "chat handled", "request_id": "r1", "latency_ms": 3.2}
    )
    entry = json.loads(JsonFormatter().format(record))
    assert entry["msg"] == "chat handled"
    assert entry["request_id"] == "r1"
    assert entry["latency_ms"] == 3.2


def test_chat_logs_request_fields_without_message_text(client):
    handler = _Collect()
    logger = logging.getLogger("veltrix")
    logger.addHandler(handler)
    try:
        resp = client.post(
            "/chat",
            json={"session_id": "s-log", "message": "my secret question"},
            headers={"X-Employee-Id": "E1001"},
        )
    finally:
        logger.removeHandler(handler)

    entries = [json.loads(line) for line in handler.lines]
    chat = next(e for e in entries if e["msg"] == "chat handled")
    assert chat["request_id"] == resp.json()["request_id"]
    assert chat["employee_id"] == "E1001"
    assert chat["route"] == "no_answer"
    assert {"tool_calls", "corpus_tokens", "latency_ms"} <= set(chat)
    assert all("my secret question" not in line for line in handler.lines)
