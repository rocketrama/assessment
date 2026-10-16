"""Which holiday calendar applies (Section 5 per region and year), end to end:
calendars → ingestion → cache → CalendarHolidays → apply_leave working days."""

import json
from datetime import date

import pytest

from app.config import Settings
from app.data.employees import Employee, get_employee
from app.data.holidays import CalendarHolidays
from app.ingestion.cache import cache_db_path
from app.ingestion.loaders.ocr import PageOcr
from app.ingestion.pipeline import run_ingestion
from app.tools.registry import ToolContext, validate_apply_leave

D = date.fromisoformat
TODAY = D("2026-09-15")


def _calendar_md(title, rows, restricted=()):
    lines = [f"# {title}", "", "## Public holidays", "", "| Date | Day | Holiday |", "| --- | --- | --- |"]
    lines += [f"| {d} | {D(d).strftime('%A')} | {n} |" for d, n in rows]
    if restricted:
        lines += ["", "## Restricted holidays", "", "| Date | Day | Holiday |", "| --- | --- | --- |"]
        lines += [f"| {d} | {D(d).strftime('%A')} | {n} |" for d, n in restricted]
    return "\n".join(lines) + "\n"


# (doc_id, regions, effective_date, access, rows, restricted)
CALENDARS = [
    ("CAL-GLOBAL-2026", ["GLOBAL"], "2026-01-01", "all", [("2026-11-02", "Global Day")], ()),
    ("CAL-IN-2025", ["IN"], "2025-01-01", "all", [("2025-11-03", "Old Day"), ("2026-11-03", "Stray 2026 Day")], ()),
    ("CAL-IN-2026", ["IN"], "2026-01-01", "all", [("2026-11-04", "India Day")], [("2026-11-05", "Optional Day")]),
    ("CAL-IN-2026-V2", ["IN"], "2026-10-01", "all", [("2026-11-06", "Later Day")], ()),
    ("CAL-DE-HR", ["DE"], "2026-01-01", "hr", [("2026-11-04", "HR-only Day")], ()),
    ("CAL-US-A", ["US"], "2026-01-01", "all", [("2026-11-03", "US Day A")], ()),
    ("CAL-US-B", ["US"], "2026-01-01", "all", [("2026-11-05", "US Day B")], ()),
]


@pytest.fixture
def cache_db(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    manifest = []
    for doc_id, regions, effective, access, rows, restricted in CALENDARS:
        (corpus / f"{doc_id}.md").write_text(_calendar_md(doc_id, rows, restricted), encoding="utf-8")
        manifest.append(
            {
                "doc_id": doc_id,
                "file": f"{doc_id}.md",
                "title": doc_id,
                "doc_type": "calendar",
                "regions": regions,
                "effective_date": effective,
                "access_level": access,
            }
        )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    settings = Settings(
        _env_file=None,
        corpus_dir=str(corpus),
        manifest_path=str(tmp_path / "manifest.json"),
        cache_dir=str(tmp_path / "cache"),
    )
    report = run_ingestion(settings, PageOcr(vision=None, use_tesseract=False))
    assert report.warnings == []
    return cache_db_path(settings.cache_dir)


def emp(region, role="employee"):
    return Employee("X1", region, role, "L3", None, "2024-01-01")


def test_region_calendar_beats_global_on_scope(cache_db):
    cal = CalendarHolidays(cache_db, emp("IN"), D("2026-01-15"))  # before V2 is effective
    resolved = cal.calendar("IN", 2026)
    assert resolved.doc_ids == ("CAL-IN-2026",)
    assert [h.name for h in resolved.holidays] == ["India Day"]  # restricted day not counted


def test_recency_and_effective_date(cache_db):
    # On 2026-09-15 the October calendar is not yet effective; later it wins on recency.
    assert CalendarHolidays(cache_db, emp("IN"), TODAY).calendar("IN", 2026).doc_ids == ("CAL-IN-2026",)
    later = CalendarHolidays(cache_db, emp("IN"), D("2026-10-01"))
    assert later.calendar("IN", 2026).doc_ids == ("CAL-IN-2026-V2",)


def test_each_year_resolved_separately(cache_db):
    cal = CalendarHolidays(cache_db, emp("IN"), TODAY)
    assert cal.calendar("IN", 2025).doc_ids == ("CAL-IN-2025",)
    assert cal.holidays("IN", D("2025-11-01"), D("2026-11-30")) == {D("2025-11-03"), D("2026-11-04")}


def test_access_level_applies_to_calendars(cache_db):
    assert CalendarHolidays(cache_db, emp("DE"), TODAY).calendar("DE", 2026).doc_ids == ("CAL-GLOBAL-2026",)
    assert CalendarHolidays(cache_db, emp("DE", "hr"), TODAY).calendar("DE", 2026).doc_ids == ("CAL-DE-HR",)


def test_conflict_combines_both_calendars(cache_db):
    resolved = CalendarHolidays(cache_db, emp("US"), TODAY).calendar("US", 2026)
    assert resolved.conflict and resolved.doc_ids == ("CAL-US-A", "CAL-US-B")
    assert {h.date for h in resolved.holidays} == {D("2026-11-03"), D("2026-11-05")}


def test_no_calendar_means_no_holidays(cache_db):
    resolved = CalendarHolidays(cache_db, emp("FR"), TODAY).calendar("FR", 2026)
    # GLOBAL still covers FR; a region with nothing at all would be empty.
    assert resolved.doc_ids == ("CAL-GLOBAL-2026",)
    assert CalendarHolidays(cache_db, emp("FR"), TODAY).calendar("FR", 2031).doc_ids == ()


def test_apply_leave_counts_working_days_with_the_resolved_calendar(cache_db, db_conn):
    requester = get_employee(db_conn, "E1001")  # IN
    ctx = ToolContext(requester, db_conn, TODAY, CalendarHolidays(cache_db, requester, TODAY))
    result = validate_apply_leave(ctx, {"leave_type": "annual", "start_date": "2026-11-02", "end_date": "2026-11-06"})
    # Mon–Fri = 5, minus India Day (4 Nov). Not minus: Global Day (loses on scope),
    # Optional Day (restricted), Later Day (not yet effective), Stray 2026 Day (2025 calendar).
    assert result.data["working_days"] == 4
