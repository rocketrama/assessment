"""Working days: Mon–Fri minus region holidays, both ends inclusive (TASK.pdf §6.2)."""

from datetime import date

import pytest

from app.tools.working_days import count_working_days
from tests.fixtures.holidays import TEST_HOLIDAYS

D = date.fromisoformat
NONE: frozenset[date] = frozenset()


@pytest.mark.parametrize(
    "start, end, holidays, expected",
    [
        ("2026-11-02", "2026-11-06", NONE, 5),  # Mon–Fri
        ("2026-11-02", "2026-11-06", frozenset({D("2026-11-04")}), 4),  # minus a holiday
        ("2026-11-07", "2026-11-08", NONE, 0),  # weekend only
        ("2026-11-06", "2026-11-09", NONE, 2),  # Fri → Mon spans a weekend
        ("2026-11-02", "2026-11-02", NONE, 1),  # single day
        ("2026-11-04", "2026-11-04", frozenset({D("2026-11-04")}), 0),  # single holiday
        ("2026-11-02", "2026-11-08", frozenset({D("2026-11-07")}), 5),  # holiday on a Saturday
        ("2026-11-02", "2026-11-13", frozenset({D("2026-11-04"), D("2026-11-10")}), 8),  # two weeks
        ("2026-11-06", "2026-11-02", NONE, 0),  # end before start
    ],
)
def test_count_working_days(start, end, holidays, expected):
    assert count_working_days(D(start), D(end), holidays) == expected


def test_holidays_are_per_region():
    start, end = D("2026-11-02"), D("2026-11-06")
    counts = {
        region: count_working_days(start, end, TEST_HOLIDAYS.holidays(region, start, end))
        for region in ("IN", "DE", "US")
    }
    assert counts == {"IN": 4, "DE": 3, "US": 5}


def test_holiday_source_only_returns_dates_in_range():
    assert TEST_HOLIDAYS.holidays("DE", D("2026-11-06"), D("2026-11-30")) == {D("2026-11-06")}
