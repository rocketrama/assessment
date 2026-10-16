"""Calendar tables → holiday rows: structure decides, restricted days stay working days."""

from datetime import date

import pytest

from app.ingestion.blocks import Block, table_block
from app.ingestion.extract_holidays import extract_holidays, parse_holiday_date

D = date.fromisoformat


@pytest.mark.parametrize(
    "text, expected",
    [
        ("15 January 2026", "2026-01-15"),
        ("15 Jan 2026", "2026-01-15"),
        ("Thursday, 15 January 2026", "2026-01-15"),
        ("January 15, 2026", "2026-01-15"),
        ("2026-01-15", "2026-01-15"),
        ("15.01.2026", "2026-01-15"),
        ("15th January 2026", "2026-01-15"),
    ],
)
def test_parse_holiday_date(text, expected):
    assert parse_holiday_date(text) == D(expected)


@pytest.mark.parametrize("text", ["Thursday", "Holiday", "31 February 2026", "15 January", ""])
def test_parse_holiday_date_rejects(text):
    assert parse_holiday_date(text) is None


def H(text, level):
    return Block("heading", text, level=level)


def table(*rows):
    return table_block([list(r) for r in rows])


PUBLIC = table(("Date", "Day", "Holiday"), ("4 November 2026", "Wednesday", "Test Day"), ("6 November 2026", "Friday", "Other Day"))
RESTRICTED = table(("Date", "Day", "Restricted holiday"), ("5 November 2026", "Thursday", "Optional Day"))


def test_public_and_restricted_tables():
    blocks = [H("Calendar", 1), H("1. Public holidays", 2), PUBLIC, H("2. Restricted holidays", 2), RESTRICTED]
    holidays, warnings = extract_holidays("CAL", ("IN",), blocks)
    assert warnings == []
    assert [(h.date.isoformat(), h.name, h.office_closed) for h in holidays] == [
        ("2026-11-04", "Test Day", True),
        ("2026-11-06", "Other Day", True),
        ("2026-11-05", "Optional Day", False),
    ]


def test_restricted_detected_from_section_heading_only():
    plain = table(("Date", "Day", "Name"), ("5 November 2026", "Thursday", "Optional Day"))
    holidays, _ = extract_holidays("CAL", ("IN",), [H("Cal", 1), H("Optional holidays", 2), plain])
    assert [h.office_closed for h in holidays] == [False]


def test_heading_scope_resets_after_restricted_section():
    blocks = [H("Cal", 1), H("Restricted", 2), RESTRICTED, H("Public", 2), PUBLIC]
    holidays, _ = extract_holidays("CAL", ("IN",), blocks)
    assert [h.office_closed for h in holidays] == [False, True, True]


def test_weekday_mismatch_is_reported():
    bad = table(("Date", "Day", "Holiday"), ("4 November 2026", "Friday", "Test Day"))
    holidays, warnings = extract_holidays("CAL", ("IN",), [bad])
    assert len(holidays) == 1
    assert warnings == ["2026-11-04 is a Wednesday, but the calendar says 'Friday'"]


def test_non_date_tables_are_ignored():
    rates = table(("Band", "Days"), ("A", "1.5"), ("B", "2.0"))
    assert extract_holidays("CAL", ("IN",), [rates]) == ([], [])


def test_one_row_per_region_and_names_without_day_column():
    two_col = table(("Date", "Holiday"), ("2026-11-04", "Test Day"))
    holidays, _ = extract_holidays("CAL", ("IN", "US"), [two_col])
    assert [(h.region, h.name) for h in holidays] == [("IN", "Test Day"), ("US", "Test Day")]


def test_unparsed_row_in_a_date_table_is_reported():
    rows = [("Date", "Holiday")] + [(f"{d} November 2026", f"Day {d}") for d in range(2, 7)] + [("TBC", "Moving day")]
    holidays, warnings = extract_holidays("CAL", ("IN",), [table(*rows)])
    assert len(holidays) == 5
    assert warnings == ["unparsed holiday date: 'TBC'"]
