"""Calendar arithmetic for future-balance questions, done in code (DESIGN §3).

Eval finding: with every input correct (balance from the tool, accrual rate and
credit day from the documents), the composer still counted the remaining
credits wrongly (answers of 10, 12 and 33 days for one question). Counting
dates is calendar work, so code does it and the composer is given the counts.

No corpus facts here (§4.1): code does not know when or how much leave is
credited. The credit day comes from the facts extracted at ingestion and
eligible for the requester (`credit_day`): month ends or month starts are
then listed alone. Eval finding: given both lists, the model added them
(7 credits instead of 4). With no clear credit-day fact, both lists are
given and the composer matches them to the documents. The rate always comes
from the documents.
"""

import calendar
import re
from datetime import date, timedelta

_MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
_MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})
_MONTHS["sept"] = 9
_MONTH = "|".join(sorted(_MONTHS, key=len, reverse=True))
_YEAR_END_RE = re.compile(r"\b(?:end of (?:the |this )?year|year[- ]end)\b", re.IGNORECASE)
_MONTH_RE = re.compile(
    rf"\b(?:by|until|till|in|at|for|end of)\s+(?:the\s+)?(?:end\s+of\s+)?({_MONTH})\b\.?(?:\s+(\d{{4}}))?",
    re.IGNORECASE,
)


def _month_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def find_target_date(message: str, today: date) -> date | None:
    """The future date a balance question is about: "by the end of December" →
    31 December; "by March" → 31 March (next year if March has passed); "by the
    end of the year" → 31 December. None when the message names no future month."""
    if _YEAR_END_RE.search(message):
        return date(today.year, 12, 31)
    m = _MONTH_RE.search(message)
    if not m:
        return None
    month = _MONTHS[m.group(1).lower()]
    year = int(m.group(2)) if m.group(2) else today.year
    target = _month_end(year, month)
    if target <= today and not m.group(2):
        target = _month_end(year + 1, month)
    return target if target > today else None


_BALANCE_RE = re.compile(r"\bbalance\b|\bhow (?:many|much)\b.*\bwill (?:i|my)\b", re.IGNORECASE)
_NOT_BALANCE_RE = re.compile(r"\bholidays?\b", re.IGNORECASE)

# Added to the search for a future-balance question: the accrual rate and credit day.
ACCRUAL_QUERY = "annual leave accrual rate per month by length of service, day the leave is credited"


def asks_future_balance(message: str, today: date) -> date | None:
    """The target date when the message asks for a balance at a future date ("how many
    days will I have by December?"), else None. Eval finding: the planner sometimes
    skipped the balance lookup or searched only carry-forward, so code ensures both."""
    if not _BALANCE_RE.search(message) or _NOT_BALANCE_RE.search(message):
        return None
    return find_target_date(message, today)


def _month_ends(after: date, until: date) -> list[date]:
    out, d = [], _month_end(after.year, after.month)
    while d <= until:
        if d > after:
            out.append(d)
        d = _month_end((d + timedelta(days=1)).year, (d + timedelta(days=1)).month)
    return out


def _month_starts(after: date, until: date) -> list[date]:
    out, d = [], date(after.year, after.month, 1)
    while d <= until:
        if d > after:
            out.append(d)
        d = date(d.year + d.month // 12, d.month % 12 + 1, 1)
    return out


def _service(join: date, on: date) -> str:
    months = (on.year - join.year) * 12 + on.month - join.month - (on.day < join.day)
    years, months = divmod(max(months, 0), 12)
    return f"{years} year{'s' * (years != 1)} {months} month{'s' * (months != 1)}"


def _anniversaries(join: date, after: date, until: date) -> list[date]:
    out = []
    for year in range(after.year, until.year + 1):
        try:
            d = join.replace(year=year)
        except ValueError:  # 29 February
            d = date(year, 3, 1)
        if after < d <= until and d > join:
            out.append(d)
    return out


_CREDIT_FACT_RE = re.compile(r"\b(?:credit\w*|accru\w*)\b.*\b(?:date|day)\b")
_LAST_DAY_RE = re.compile(r"\b(?:last (?:calendar |working )?day|end of (?:the |each )?month|month[- ]end)\b")
_FIRST_DAY_RE = re.compile(r"\b(?:first (?:calendar |working )?day|1st|start of (?:the |each )?month)\b")


def credit_day(facts) -> str | None:
    """"end" or "start" when the requester's eligible facts name one monthly credit day for
    leave other than sick leave; None when none does or they disagree."""
    kinds = set()
    for fact in facts:
        key = f"{fact.topic} {fact.attribute}".lower()
        if "sick" in key or not _CREDIT_FACT_RE.search(key):
            continue
        value = fact.value.lower()
        if _LAST_DAY_RE.search(value):
            kinds.add("end")
        elif _FIRST_DAY_RE.search(value):
            kinds.add("start")
    return kinds.pop() if len(kinds) == 1 else None


def calendar_note(today: date, target: date, join_date: str, credit: str | None = None) -> str:
    """The note given to the composer: dates and counts only, no rule. `credit` ("end" or
    "start", from `credit_day`) lists only the dates matching the documents' credit day."""
    ends = _month_ends(today, target)
    starts = _month_starts(today, target)
    join = date.fromisoformat(join_date)

    def listing(ds: list[date]) -> str:
        return f"{len(ds)} ({', '.join(d.isoformat() for d in ds)})" if ds else "0"

    period = f"from today {today.isoformat()} (excluded) to {target.isoformat()} (included)"
    if credit in ("end", "start"):
        day = "last day of the month" if credit == "end" else "first day of the month"
        dates = f"there are {listing(ends if credit == 'end' else starts)} credit dates on the {day}, the credit day the documents give."
    else:
        dates = f"there are {listing(ends)} month-end dates and {listing(starts)} first days of a month."
    parts = [
        f"Future-balance calendar, computed in code (use these counts; do not count dates yourself): {period} {dates}",
        f"Length of service (joined {join.isoformat()}): {_service(join, today)} today, "
        f"{_service(join, target)} on {target.isoformat()}.",
    ]
    if anniversaries := _anniversaries(join, today, target):
        parts.append("Service anniversary in this period: " + ", ".join(d.isoformat() for d in anniversaries) + ".")
    parts.append(
        "Future balance = current balance from the tool results + (number of credit dates "
        + ("above" if credit in ("end", "start") else "matching the credit day the documents give")
        + " × the accrual rate for this length of service), then apply any balance limit. "
        "State the date used, each number and the result."
    )
    return " ".join(parts)
