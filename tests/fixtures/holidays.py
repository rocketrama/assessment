"""In-memory HolidaySource for tests. The dates are invented, not from the corpus."""

from collections.abc import Iterable, Mapping
from datetime import date


class StaticHolidays:
    def __init__(self, by_region: Mapping[str, Iterable[str]] | None = None) -> None:
        self._by_region = {
            region: frozenset(date.fromisoformat(d) for d in days)
            for region, days in (by_region or {}).items()
        }

    def holidays(self, region: str, start: date, end: date) -> frozenset[date]:
        return frozenset(d for d in self._by_region.get(region, ()) if start <= d <= end)


# 2026-11-02 is a Monday. One fake IN holiday on Wed 4 Nov, two fake DE ones on 5–6 Nov.
TEST_HOLIDAYS = StaticHolidays({"IN": ["2026-11-04"], "DE": ["2026-11-05", "2026-11-06"]})
