"""Working days (TASK.pdf §6.2): dates in [start, end] that fall Monday–Friday and are
not public holidays for the requester's region.

Pure: the holidays come in from a `HolidaySource`. The real source (Step 4) reads
the `holidays` table extracted at ingestion from the calendar document that
applies under Section 5; nothing here knows any holiday date.
"""

from datetime import date, timedelta
from typing import Protocol


class HolidaySource(Protocol):
    def holidays(self, region: str, start: date, end: date) -> frozenset[date]:
        """Public holidays for `region` between `start` and `end`, inclusive."""
        ...


def count_working_days(start: date, end: date, holidays: frozenset[date]) -> int:
    """Both ends inclusive; 0 when `end` is before `start`."""
    days = (end - start).days + 1
    return sum(
        1
        for offset in range(max(days, 0))
        if (d := start + timedelta(days=offset)).weekday() < 5 and d not in holidays
    )
