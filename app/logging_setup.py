"""Structured logging for the `veltrix` logger tree.

Fields passed via `extra=` (request_id, employee_id, route, latency_ms, ...) become
top-level JSON keys. Callers log IDs only, never names, emails or phones (DESIGN §5);
redaction of free text is added with `privacy/log_filter.py`.
"""

import json
import logging
import sys
from datetime import datetime, timezone

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
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = f"{record.levelname:<7} {record.name}: {record.getMessage()}"
        extras = " ".join(f"{k}={v}" for k, v in vars(record).items() if k not in _RESERVED)
        text = f"{base} {extras}" if extras else base
        if record.exc_info:
            text += "\n" + self.formatException(record.exc_info)
        return text



def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Idempotent: replaces the handler, so repeated `create_app()` calls don't duplicate lines."""
    logger = logging.getLogger("veltrix")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(TextFormatter() if fmt == "text" else JsonFormatter())
    logger.handlers = [handler]
    logger.setLevel(level.upper())
    # Own handler; don't also print through uvicorn's root configuration.
    logger.propagate = False
