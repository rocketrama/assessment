"""Email masking and phone stripping (DESIGN §5)."""

import pytest

from app.privacy.masking import PHONE_REMOVED, mask_email, mask_emails, redact_contacts, strip_phones


@pytest.mark.parametrize(
    "email, masked",
    [
        ("e.in@veltrix.example", "e***@veltrix.example"),
        ("R@x.io", "R***@x.io"),
        ("  hr.one@veltrix.example ", "h***@veltrix.example"),
        ("not-an-email", "***"),
        ("@veltrix.example", "***"),
        ("", "***"),
    ],
)
def test_mask_email(email, masked):
    assert mask_email(email) == masked


def test_mask_emails_in_text():
    text = "Write to e.in@veltrix.example or hr.one@veltrix.example."
    assert mask_emails(text) == "Write to e***@veltrix.example or h***@veltrix.example."


@pytest.mark.parametrize(
    "phone",
    ["+00-000-0004", "+1 (415) 555-0100", "+91 98765 43210", "98765 43210", "9876543210", "415.555.0100"],
)
def test_strip_phones_removes_phone_formats(phone):
    assert strip_phones(f"Call {phone} today") == f"Call {PHONE_REMOVED} today"


@pytest.mark.parametrize(
    "text",
    [
        "Leave from 2026-10-05 to 2026-10-07",
        "Request REQ-FIX-0001 is approved",
        "Employee E1001 has 12.5 days",
        "Band max is 1500000 per year",
        "Holidays: 2026-12-25, 2026-12-26",
    ],
)
def test_strip_phones_keeps_dates_ids_and_amounts(text):
    assert strip_phones(text) == text


def test_redact_contacts_handles_both():
    text = "Reach e.in@veltrix.example on +00-000-0004."
    assert redact_contacts(text) == f"Reach e***@veltrix.example on {PHONE_REMOVED}."
