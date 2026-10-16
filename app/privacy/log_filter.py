"""PII redaction for every `veltrix` log line (DESIGN §5).

Callers already log IDs only; this filter is the safety net for free text that
slips in (exception messages, a stray f-string). It runs on the handler, so it
covers records from every child logger, and rewrites the record before any
formatter sees it.
"""

import logging
import re
from collections.abc import Iterable

from app.privacy.masking import redact_contacts

NAME_REDACTED = "[name]"

# Attributes every LogRecord has; anything else came from `extra=`.
_RESERVED = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class PiiLogFilter(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self._names_re: re.Pattern[str] | None = None

    def set_names(self, names: Iterable[str]) -> None:
        """Full names to redact (from the employees table). Longest first, so
        "Test Employee In Two" is not cut to "[name] Two"."""
        cleaned = sorted({n.strip() for n in names if n and n.strip()}, key=len, reverse=True)
        self._names_re = (
            re.compile(r"\b(?:" + "|".join(map(re.escape, cleaned)) + r")\b", re.IGNORECASE)
            if cleaned
            else None
        )

    def redact(self, text: str) -> str:
        text = redact_contacts(text)
        if self._names_re is not None:
            text = self._names_re.sub(NAME_REDACTED, text)
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self.redact(record.getMessage())
        record.args = None
        if record.exc_info:
            # Formatters print exc_text; the traceback is rendered once, redacted.
            record.exc_text = self.redact(logging.Formatter().formatException(record.exc_info))
            record.exc_info = None
        for key, value in list(vars(record).items()):
            if key not in _RESERVED and isinstance(value, str):
                setattr(record, key, self.redact(value))
        return True


# One shared instance: logging_setup attaches it, startup loads the names.
PII_FILTER = PiiLogFilter()
