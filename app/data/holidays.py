"""The holiday calendar that applies to a requester (TASK.pdf §6.2, §5.1).

Implements tools/working_days.py::HolidaySource over the `holidays` table that
ingestion fills from `calendar` documents. Which calendar counts is decided by
the Section 5 rules, never by file name: for each region and calendar year,
among the calendars that are eligible for the requester on the reference date
(readable, for their region or GLOBAL, already effective) and list dates in that
year, the precedence winner (scope, then recency) supplies the holidays. Equal
winners are an unresolved conflict: their dates are combined and the conflict is
reported, so no listed closure is silently ignored.

Only rows with office_closed=1 are holidays; restricted/optional days are not.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import date

from app.data.employees import Employee
from app.retrieval.eligibility import DocMeta, is_eligible
from app.retrieval.precedence import winners

log = logging.getLogger("veltrix.holidays")


@dataclass(frozen=True)
class HolidayRow:
    doc_id: str
    region: str
    date: date
    name: str
    office_closed: bool


@dataclass(frozen=True)
class ResolvedCalendar:
    year: int
    doc_ids: tuple[str, ...]  # winners; more than one = unresolved conflict
    holidays: tuple[HolidayRow, ...]  # office-closed days of the winners in that year

    @property
    def conflict(self) -> bool:
        return len(self.doc_ids) > 1


def _load(conn: sqlite3.Connection) -> tuple[dict[str, DocMeta], list[HolidayRow]]:
    docs = {
        r["doc_id"]: DocMeta(
            r["doc_id"],
            r["doc_type"],
            tuple(json.loads(r["regions_json"])),
            date.fromisoformat(r["effective_date"]),
            r["access_level"],
        )
        for r in conn.execute("SELECT * FROM documents WHERE doc_type = 'calendar'")
    }
    rows = [
        HolidayRow(r["doc_id"], r["region"], date.fromisoformat(r["date"]), r["name"], bool(r["office_closed"]))
        for r in conn.execute("SELECT * FROM holidays ORDER BY date, doc_id")
    ]
    return docs, rows


class CalendarHolidays:
    """HolidaySource for one requester. Reads the cache DB on each call, so a
    re-ingested corpus is picked up without restarting tool code."""

    def __init__(self, cache_db: str, requester: Employee, reference_date: date) -> None:
        self._cache_db = cache_db
        self._role = requester.role
        self._reference_date = reference_date

    def calendar(self, region: str, year: int) -> ResolvedCalendar:
        conn = sqlite3.connect(f"file:{self._cache_db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            docs, rows = _load(conn)
        finally:
            conn.close()

        # Region is decided per document by eligibility (a GLOBAL calendar's rows are
        # stored under "GLOBAL"), so rows are matched by document and year only.
        listing_year = {h.doc_id for h in rows if h.date.year == year}
        eligible = [
            d for d in docs.values()
            if d.doc_id in listing_year and is_eligible(d, self._role, region, self._reference_date)
        ]
        top = winners(eligible)
        doc_ids = tuple(d.doc_id for d in top)
        if len(doc_ids) > 1:
            log.warning("holiday calendars conflict", extra={"region": region, "year": year, "doc_ids": list(doc_ids)})
        if not doc_ids:
            log.warning("no holiday calendar applies", extra={"region": region, "year": year})
        # A calendar for several regions stores each date once per region; keep one.
        holidays: dict[tuple[str, date], HolidayRow] = {}
        for h in rows:
            if h.doc_id in doc_ids and h.date.year == year and h.office_closed:
                holidays.setdefault((h.doc_id, h.date), h)
        return ResolvedCalendar(year, doc_ids, tuple(holidays.values()))

    def holidays(self, region: str, start: date, end: date) -> frozenset[date]:
        out: set[date] = set()
        for year in range(start.year, end.year + 1):
            out.update(h.date for h in self.calendar(region, year).holidays if start <= h.date <= end)
        return frozenset(out)
