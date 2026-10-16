"""Structured logging for the `veltrix` logger tree.

Fields passed via `extra=` (request_id, employee_id, route, latency_ms, ...) become
top-level JSON keys. Callers log IDs only, never names, emails or phones (DESIGN §5);
`privacy/log_filter.py` redacts any free text that slips through.
"""

import json
import logging
import sys
from datetime import datetime, timezone

from app.privacy.log_filter import PII_FILTER

# Attributes every LogRecord has; anything else came from `extra=`.
_RESERVED = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        entry.update({k: v for k, v in vars(record).items() if k not in _RESERVED})
        exc = _exception_text(self, record)
        if exc:
            entry["exc"] = exc
        return json.dumps(entry, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = f"{record.levelname:<7} {record.name}: {record.getMessage()}"
        extras = " ".join(f"{k}={v}" for k, v in vars(record).items() if k not in _RESERVED)
        text = f"{base} {extras}" if extras else base
        exc = _exception_text(self, record)
        if exc:
            text += "\n" + exc
        return text


def _exception_text(formatter: logging.Formatter, record: logging.LogRecord) -> str | None:
    # The PII filter pre-renders tracebacks into exc_text (redacted) and clears exc_info.
    if record.exc_text:
        return record.exc_text
    if record.exc_info:
        return formatter.formatException(record.exc_info)
    return None


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Idempotent: replaces the handler, so repeated `create_app()` calls don't duplicate lines."""
    logger = logging.getLogger("veltrix")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(TextFormatter() if fmt == "text" else JsonFormatter())
    handler.addFilter(PII_FILTER)
    logger.handlers = [handler]
    logger.setLevel(level.upper())
    # Own handler; don't also print through uvicorn's root configuration.
    logger.propagate = False
