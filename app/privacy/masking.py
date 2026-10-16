"""PII masking (DESIGN §5): emails leave masked as `r***@domain`, phone numbers never leave.

Pure functions with no dependencies, shared by the data layer, the response
validator and the log filter.
"""

import re

PHONE_REMOVED = "[phone removed]"

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# A run of digits joined by phone separators, not touching letters/digits/hyphens,
# so IDs like REQ-FIX-0001 or E1001 never match as a whole.
_PHONE_CANDIDATE_RE = re.compile(r"(?<![\w-])\+?\(?\d[\d\s().-]{5,}\d\)?(?![\w-])")
_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_SEPARATOR_RE = re.compile(r"\d[\s().-]+\d")


def mask_email(email: str) -> str:
    """`e.in@veltrix.example` → `e***@veltrix.example`. Anything that isn't an email → `***`."""
    local, at, domain = email.strip().partition("@")
    if not at or not local or not domain:
        return "***"
    return f"{local[0]}***@{domain}"


def mask_emails(text: str) -> str:
    return EMAIL_RE.sub(lambda m: mask_email(m.group()), text)


def _is_phone(candidate: str) -> bool:
    digits = sum(c.isdigit() for c in candidate)
    if not 7 <= digits <= 15 or _ISO_DATE_RE.search(candidate):
        return False
    # Plain 7–9 digit runs are more likely amounts than phones; keep them.
    return candidate.startswith("+") or bool(_SEPARATOR_RE.search(candidate)) or digits >= 10


def strip_phones(text: str) -> str:
    return _PHONE_CANDIDATE_RE.sub(
        lambda m: PHONE_REMOVED if _is_phone(m.group()) else m.group(), text
    )


def redact_contacts(text: str) -> str:
    """Emails masked, phones removed — for any free text leaving the service."""
    return strip_phones(mask_emails(text))
