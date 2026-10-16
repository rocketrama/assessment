"""Holiday calendar tables → holiday rows (TASK.pdf §6.2, ARCHITECTURE §5.6).

Structure, not content, decides what is a holiday: any table in a `calendar`
document whose cells are mostly dates is a holiday list. A table whose section
heading or header row says restricted / optional / floating lists days the
office stays open, so its rows are stored with `office_closed=False` and never
reduce working days. When a weekday column is present, each date is checked
against it and mismatches are reported. Nothing about specific holidays is
written in code (§4.1).
"""

import re
from dataclasses import dataclass
from datetime import date, datetime

from app.ingestion.blocks import Block

_OPTIONAL_RE = re.compile(r"\b(restricted|optional|floating|elective)\b", re.IGNORECASE)
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_DATE_FORMATS = ("%Y-%m-%d", "%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y", "%d.%m.%Y", "%d/%m/%Y")
MIN_DATE_SHARE = 0.8  # share of body cells in a column that must parse as dates


@dataclass(frozen=True)
class Holiday:
    doc_id: str
    region: str
    date: date
    name: str
    office_closed: bool


def parse_holiday_date(text: str) -> date | None:
    """'15 January 2026', 'Thursday, 1 January 2026', 'January 1, 2026', '2026-01-01', '01.01.2026'."""
    cleaned = re.sub(r"\s+", " ", text.replace(",", " ")).strip()
    words = cleaned.split(" ")
    if words and words[0].lower().rstrip(".") in _WEEKDAYS + [d[:3] for d in _WEEKDAYS]:
        cleaned = " ".join(words[1:])
    cleaned = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", cleaned)
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


def _column_share(rows: tuple[tuple[str, ...], ...], col: int, test) -> float:
    cells = [r[col] for r in rows if col < len(r) and r[col].strip()]
    return sum(1 for c in cells if test(c)) / len(cells) if cells else 0.0


def extract_holidays(
    doc_id: str, regions: tuple[str, ...], blocks: list[Block]
) -> tuple[list[Holiday], list[str]]:
    holidays: list[Holiday] = []
    warnings: list[str] = []
    heading_by_level: dict[int, str] = {}

    for block in blocks:
        if block.kind == "heading":
            heading_by_level = {k: v for k, v in heading_by_level.items() if k < block.level}
            heading_by_level[block.level] = block.text
            continue
        if block.kind != "table" or len(block.rows) < 2:
            continue
        header, body = block.rows[0], block.rows[1:]
        width = len(header)
        date_cols = [c for c in range(width) if _column_share(body, c, parse_holiday_date) >= MIN_DATE_SHARE]
        if not date_cols:
            continue
        date_col = date_cols[0]
        day_cols = [
            c for c in range(width)
            if c != date_col and _column_share(body, c, lambda s: s.strip().lower() in _WEEKDAYS) >= MIN_DATE_SHARE
        ]
        name_cols = [c for c in range(width) if c != date_col and c not in day_cols]
        section = " ".join(heading_by_level.values())
        office_closed = not (_OPTIONAL_RE.search(section) or _OPTIONAL_RE.search(" ".join(header)))

        for row in body:
            parsed = parse_holiday_date(row[date_col]) if date_col < len(row) else None
            if parsed is None:
                if any(cell.strip() for cell in row):
                    warnings.append(f"unparsed holiday date: {row[date_col]!r}")
                continue
            if day_cols and row[day_cols[0]].strip().lower() != _WEEKDAYS[parsed.weekday()]:
                warnings.append(
                    f"{parsed.isoformat()} is a {_WEEKDAYS[parsed.weekday()].title()}, "
                    f"but the calendar says {row[day_cols[0]]!r}"
                )
            name = " ".join(row[c] for c in name_cols if c < len(row) and row[c].strip())
            for region in regions:
                holidays.append(Holiday(doc_id, region, parsed, name, office_closed))
    return holidays, warnings
