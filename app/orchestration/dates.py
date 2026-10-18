"""Date phrases → calendar dates, deterministically (TASK.pdf §8.4).

The planner copies date phrases from the message; this module, not the LLM,
turns them into dates against APP_TODAY:
- a date without a year is in 2026 (the leave year);
- weeks start on Monday; "next week" is Monday–Friday of the following week;
- "next Friday" is the first Friday after today (so is "Friday" / "this Friday");
- anything ambiguous ("03/04", "this week", today's own weekday name) or not
  understood raises DateError, and the caller asks the user (route clarify).
"""

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta

DEFAULT_YEAR = 2026

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
_MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})
_MONTHS["sept"] = 9


class DateError(Exception):
    """The phrase can't be turned into one date: the user must be asked."""

    def __init__(self, phrase: str, reason: str) -> None:
        super().__init__(f"{phrase!r}: {reason}")
        self.phrase = phrase
        self.reason = reason  # "ambiguous" | "not_understood" | "past"


@dataclass(frozen=True)
class DateRange:
    start: date
    end: date


def _clean(phrase: str) -> str:
    text = phrase.lower().strip()
    text = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", text)
    text = re.sub(r"[,]", " ", text)
    text = re.sub(r"\b(the|of|on|from|until|till|to)\b", " ", text)
    return " ".join(text.split())


_WEEKDAY_YEAR_RE = re.compile(r"((?:(?:next|this) )?(?:" + "|".join(_WEEKDAYS) + r")) (\d{4})")


def _split_weekday_year(text: str) -> tuple[str, int | None]:
    """'friday 2026' → ('friday', 2026). Eval finding: the planner added the year to
    "until Friday". The year then only has to agree with the weekday's date."""
    if m := _WEEKDAY_YEAR_RE.fullmatch(text):
        return m[1], int(m[2])
    return text, None


def _check_year(result: date, year: int | None, phrase: str) -> date:
    if year is not None and result.year != year:
        raise DateError(phrase, "not_understood")
    return result


def _next_weekday(today: date, weekday: int) -> date:
    days = (weekday - today.weekday()) % 7 or 7
    return today + timedelta(days=days)


def _explicit(text: str) -> date | None:
    """ISO, '21 september [2026]', 'september 21 [2026]', '21/09/2026', '21.9';
    a leading weekday name ('monday 21 september 2026') is ignored."""
    words = text.split(" ")
    if len(words) > 1 and words[0] in _WEEKDAYS:
        text = " ".join(words[1:])
    if m := re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", text):
        return date(int(m[1]), int(m[2]), int(m[3]))
    if m := re.fullmatch(r"(\d{1,2}) ([a-z]+)(?: (\d{4}))?", text):
        if m[2] in _MONTHS:
            return date(int(m[3] or DEFAULT_YEAR), _MONTHS[m[2]], int(m[1]))
    if m := re.fullmatch(r"([a-z]+) (\d{1,2})(?: (\d{4}))?", text):
        if m[1] in _MONTHS:
            return date(int(m[3] or DEFAULT_YEAR), _MONTHS[m[1]], int(m[2]))
    if m := re.fullmatch(r"(\d{1,2})[/.](\d{1,2})(?:[/.](\d{4}))?", text):
        a, b = int(m[1]), int(m[2])
        if a <= 12 and b <= 12 and a != b:
            raise ValueError("ambiguous")  # 03/04: March 4th or 3 April?
        day, month = (a, b) if a > 12 or a == b else (b, a) if b > 12 else (a, b)
        return date(int(m[3] or DEFAULT_YEAR), month, day)
    return None


def resolve_date(phrase: str, today: date, allow_past: bool = True) -> date:
    text, year = _split_weekday_year(_clean(phrase))
    if not text:
        raise DateError(phrase, "not_understood")
    if text == "today":
        result = today
    elif text == "tomorrow":
        result = today + timedelta(days=1)
    elif m := re.fullmatch(r"(?:(next|this) )?(" + "|".join(_WEEKDAYS) + r")", text):
        weekday = _WEEKDAYS.index(m[2])
        if m[1] != "next" and weekday == today.weekday():
            raise DateError(phrase, "ambiguous")  # "Tuesday" said on a Tuesday
        result = _check_year(_next_weekday(today, weekday), year, phrase)
    else:
        try:
            result = _explicit(text)
        except ValueError as exc:
            reason = "ambiguous" if str(exc) == "ambiguous" else "not_understood"
            raise DateError(phrase, reason) from None
        if result is None:
            raise DateError(phrase, "not_understood")
    if not allow_past and result < today:
        raise DateError(phrase, "past")
    return result


def resolve_range(start_phrase: str, end_phrase: str | None, today: date, allow_past: bool = True) -> DateRange:
    """One range from a start and optional end phrase. 'next week' (as the start,
    or as both start and end) is Monday–Friday of the following week; a single
    day is a one-day range."""
    phrases = {_clean(start_phrase), _clean(end_phrase or "next week")}
    if phrases == {"next week"}:
        monday = today + timedelta(days=7 - today.weekday())
        return DateRange(monday, monday + timedelta(days=4))
    end = resolve_date(end_phrase, today, allow_past) if end_phrase else None
    if end is not None and re.fullmatch(r"\d{1,2}", _clean(start_phrase)):
        # "from 21 to 23 September": the start borrows the end's month and year.
        try:
            start = end.replace(day=int(_clean(start_phrase)))
        except ValueError:
            raise DateError(start_phrase, "not_understood") from None
        if not allow_past and start < today:
            raise DateError(start_phrase, "past")
    else:
        start = resolve_date(start_phrase, today, allow_past)
    end = end or start
    if end < start and not re.search(r"\d{4}", end_phrase or ""):
        # "28 December to 2 January" without years is outside the leave year.
        raise DateError(end_phrase or start_phrase, "ambiguous")
    return DateRange(start, end)


def resolve_end(phrase: str, start: date, today: date) -> date:
    """End of a leave range. A bare weekday ("until Friday") means the first such
    day on or after the start — for leave starting Monday 21 September, "Friday"
    is the 25th, not the Friday before it. Other phrases resolve as usual."""
    text, year = _split_weekday_year(_clean(phrase))
    if m := re.fullmatch(r"(?:this )?(" + "|".join(_WEEKDAYS) + r")", text):
        return _check_year(start + timedelta(days=(_WEEKDAYS.index(m[1]) - start.weekday()) % 7), year, phrase)
    return resolve_date(phrase, today)


def resolve_reference_date(phrase: str | None, today: date) -> date | None:
    """The past date a policy question is about (§5), or None for today.

    "2025" / "in 2025" → 31 Dec 2025; "March 2025" → 31 March 2025; an explicit
    past date → that date; "last year" → 31 Dec of last year. Anything that is
    not in the past → None (today applies; future documents are announced as
    upcoming changes instead)."""
    if not phrase:
        return None
    text = _clean(re.sub(r"\b(in|during|as at|as of|back)\b", " ", phrase.lower()))
    if text == "last year":
        return date(today.year - 1, 12, 31)
    if m := re.fullmatch(r"(\d{4})", text):
        result = date(int(m[1]), 12, 31)
    elif (m := re.fullmatch(r"([a-z]+) (\d{4})", text)) and m[1] in _MONTHS:
        month = _MONTHS[m[1]]
        result = date(int(m[2]), month, calendar.monthrange(int(m[2]), month)[1])
    else:
        try:
            result = resolve_date(text, today)
        except DateError:
            return None
    return result if result < today else None


_PHRASE_RE = re.compile(
    r"\b(?:next week|today|tomorrow|(?:next |this )?(?:" + "|".join(_WEEKDAYS) + r")"
    r"|\d{4}-\d{1,2}-\d{1,2}"
    r"|\d{1,2}(?:st|nd|rd|th)? (?:of )?(?:" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\b(?: \d{4})?"
    r"|(?:" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r") \d{1,2}(?:st|nd|rd|th)?(?:,? \d{4})?)",
    re.IGNORECASE,
)


def find_date_phrases(message: str) -> list[str]:
    """Date phrases in a message, in order — the fallback when the planner left
    the dates out although the user gave them."""
    return [m.group(0) for m in _PHRASE_RE.finditer(message)]


_DAY = r"\d{1,2}(?:st|nd|rd|th)?"
_MONTH = "(?:" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + ")"
_FULL = rf"(?:{_DAY} (?:of )?{_MONTH}\b(?: \d{{4}})?|{_MONTH} {_DAY}(?:,? \d{{4}})?|\d{{4}}-\d{{1,2}}-\d{{1,2}})"
_RANGE_RE = re.compile(
    rf"(?:\bfrom |\bbetween )?\b({_FULL}|{_DAY})\s*(?:-|\u2013|\u2014|\bto\b|\band\b|\buntil\b|\btill\b|\bthrough\b)\s*({_FULL})",
    re.IGNORECASE,
)


def find_date_range(message: str) -> tuple[str, str] | None:
    """An explicit range written in the message ("from 21 to 23 September",
    "between 5 and 9 October", "21-23 Sept") as (start phrase, end phrase).
    The message is the ground truth: this beats the planner's copy of it."""
    m = _RANGE_RE.search(message)
    return (m.group(1), m.group(2)) if m else None


# A month name alone needs "in" / "during" / … before it ("may" is also a verb).
_MONTH_PERIOD_RE = re.compile(
    rf"\b(?:(next|this) month|(?:in|during|for|throughout) ({_MONTH})\b(?:,? (\d{{4}}))?)", re.IGNORECASE
)


def find_month_period(message: str, today: date) -> DateRange | None:
    """A whole month named in the message: "next month", "this month", "in October
    [2026]" → its first to last day; a month without a year is in 2026. For team
    leave lookups only: a leave request for "next month" still needs exact dates.
    Live finding: "Is anyone in my team off next month?" was answered with a question."""
    m = _MONTH_PERIOD_RE.search(message)
    if not m:
        return None
    if m[1]:
        year, month = today.year, today.month
        if m[1].lower() == "next":
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    else:
        year, month = int(m[3] or DEFAULT_YEAR), _MONTHS[m[2].lower()]
    return DateRange(date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1]))
