"""Every `veltrix` log line is scrubbed of emails, phones and employee names (DESIGN §5)."""

import json
import logging

from app.logging_setup import configure_logging
from app.privacy.log_filter import NAME_REDACTED, PII_FILTER, PiiLogFilter
from app.privacy.masking import PHONE_REMOVED


def _filtered(record_fields: dict, names=("Test Employee In", "Test Employee In Two")):
    f = PiiLogFilter()
    f.set_names(names)
    record = logging.makeLogRecord({"name": "veltrix", "levelname": "INFO", **record_fields})
    assert f.filter(record) is True
    return record


def test_message_and_args_are_redacted():
    record = _filtered(
        {"msg": "lookup %s at %s, phone %s", "args": ("Test Employee In", "e.in@veltrix.example", "+00-000-0004")}
    )
    assert record.getMessage() == f"lookup {NAME_REDACTED} at e***@veltrix.example, phone {PHONE_REMOVED}"


def test_longest_name_wins_and_match_is_case_insensitive():
    record = _filtered({"msg": "TEST EMPLOYEE IN TWO and test employee in"})
    assert record.getMessage() == f"{NAME_REDACTED} and {NAME_REDACTED}"


def test_string_extras_are_redacted_ids_kept():
    record = _filtered({"msg": "x", "detail": "mail e.in@veltrix.example", "employee_id": "E1001", "latency_ms": 3.0})
    assert record.detail == "mail e***@veltrix.example"
    assert record.employee_id == "E1001"
    assert record.latency_ms == 3.0


def test_traceback_is_redacted():
    try:
        raise ValueError("bad row for Test Employee In, e.in@veltrix.example")
    except ValueError:
        import sys

        record = _filtered({"msg": "failed", "exc_info": sys.exc_info()})
    assert record.exc_info is None
    assert "Test Employee In" not in record.exc_text
    assert "e.in@veltrix.example" not in record.exc_text
    assert "ValueError" in record.exc_text


def test_configured_logger_output_has_no_pii(capsys):
    PII_FILTER.set_names(["Test Employee In"])
    try:
        configure_logging("INFO", "json")
        log = logging.getLogger("veltrix.child")
        log.info("hello Test Employee In <e.in@veltrix.example> +00-000-0004", extra={"employee_id": "E1001"})
        try:
            raise RuntimeError("Test Employee In")
        except RuntimeError:
            log.exception("boom")
    finally:
        PII_FILTER.set_names([])

    out = capsys.readouterr().out
    entries = [json.loads(line) for line in out.splitlines()]
    assert entries[0]["msg"] == f"hello {NAME_REDACTED} <e***@veltrix.example> {PHONE_REMOVED}"
    assert entries[0]["employee_id"] == "E1001"
    assert "RuntimeError" in entries[1]["exc"]
    for secret in ("Test Employee In", "e.in@veltrix.example", "+00-000-0004"):
        assert secret not in out
