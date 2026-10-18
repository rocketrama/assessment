"""Date phrases (TASK.pdf §8.4). APP_TODAY in tests is Tuesday 2026-09-15."""

from datetime import date

import pytest

from app.orchestration.dates import (
    DateError,
    find_month_period,
    resolve_date,
    resolve_end,
    resolve_range,
    resolve_reference_date,
)

TODAY = date(2026, 9, 15)  # Tuesday
D = date.fromisoformat


@pytest.mark.parametrize(
    "phrase, expected",
    [
        ("2026-09-21", "2026-09-21"),
        ("21 September", "2026-09-21"),  # no year → 2026
        ("21st of September", "2026-09-21"),
        ("September 21", "2026-09-21"),
        ("Sept 21, 2026", "2026-09-21"),
        ("21/09/2026", "2026-09-21"),
        ("21.9", "2026-09-21"),
        ("today", "2026-09-15"),
        ("tomorrow", "2026-09-16"),
        ("next Friday", "2026-09-18"),  # first Friday after today
        ("Friday", "2026-09-18"),
        ("this Friday", "2026-09-18"),
        ("next Tuesday", "2026-09-22"),  # today's weekday with "next" → a week on
        ("next Monday", "2026-09-21"),
    ],
)
def test_resolve_date(phrase, expected):
    assert resolve_date(phrase, TODAY) == D(expected)


@pytest.mark.parametrize(
    "phrase, reason",
    [
        ("03/04", "ambiguous"),  # 3 April or March 4th?
        ("Tuesday", "ambiguous"),  # said on a Tuesday: today or next week?
        ("soon", "not_understood"),
        ("the week after Diwali", "not_understood"),
        ("31 February", "not_understood"),
        ("", "not_understood"),
    ],
)
def test_resolve_date_asks_when_unclear(phrase, reason):
    with pytest.raises(DateError) as exc:
        resolve_date(phrase, TODAY)
    assert exc.value.reason == reason


def test_past_dates_rejected_only_when_asked():
    assert resolve_date("1 September", TODAY) == D("2026-09-01")
    with pytest.raises(DateError) as exc:
        resolve_date("1 September", TODAY, allow_past=False)
    assert exc.value.reason == "past"


def span(r):
    return (r.start, r.end)


def test_next_week_is_monday_to_friday_of_the_following_week():
    assert span(resolve_range("next week", None, TODAY)) == (D("2026-09-21"), D("2026-09-25"))
    assert span(resolve_range("next week", "next week", TODAY)) == (D("2026-09-21"), D("2026-09-25"))
    sunday = date(2026, 9, 20)
    assert resolve_range("next week", None, sunday).start == D("2026-09-21")


def test_ranges():
    assert span(resolve_range("21 September", "25 September", TODAY)) == (D("2026-09-21"), D("2026-09-25"))
    assert span(resolve_range("next Monday", None, TODAY)) == (D("2026-09-21"), D("2026-09-21"))
    # "Friday" is the 18th, before the 21st: the user must say what they mean.
    with pytest.raises(DateError):
        resolve_range("21 September", "Friday", TODAY)


@pytest.mark.parametrize(
    "start, end, expected",
    [
        ("21", "23 September", ("2026-09-21", "2026-09-23")),  # "from 21 to 23 September"
        ("1", "30 October", ("2026-10-01", "2026-10-30")),
        ("5", "9 October", ("2026-10-05", "2026-10-09")),
        ("Monday 21 September 2026", "Friday", ("2026-09-21", "2026-09-25")),
    ],
)
def test_ranges_as_people_write_them(start, end, expected):
    from app.orchestration.dates import resolve_end

    if end == "Friday":  # end phrases relative to the start go through resolve_end
        s = resolve_date(start, TODAY)
        assert (s.isoformat(), resolve_end(end, s, TODAY).isoformat()) == expected
    else:
        r = resolve_range(start, end, TODAY)
        assert (r.start.isoformat(), r.end.isoformat()) == expected


def test_bare_day_without_a_month_alone_is_not_understood():
    with pytest.raises(DateError):
        resolve_range("21", None, TODAY)


def test_range_ending_before_start_without_years_is_ambiguous():
    with pytest.raises(DateError):
        resolve_range("28 December", "2 January", TODAY)


@pytest.mark.parametrize(
    "phrase, expected",
    [
        ("2025", "2025-12-31"),
        ("in 2025", "2025-12-31"),
        ("last year", "2025-12-31"),
        ("March 2025", "2025-03-31"),
        ("1 August 2025", "2025-08-01"),
        ("2026", None),  # current year → today applies
        ("2027", None),  # future → upcoming changes, not a reference date
        ("next year", None),
        (None, None),
    ],
)
def test_reference_date(phrase, expected):
    assert resolve_reference_date(phrase, TODAY) == (D(expected) if expected else None)


@pytest.mark.parametrize(
    "message, phrases",
    [
        ("Who in my team is on leave next week?", ["next week"]),
        ("Anyone off from 21 September to 25th September 2026?", ["21 September", "25th September 2026"]),
        ("team leave next Friday please", ["next Friday"]),
        ("between 2026-10-01 and 2026-10-05", ["2026-10-01", "2026-10-05"]),
        ("What's the policy?", []),
    ],
)
def test_find_date_phrases(message, phrases):
    from app.orchestration.dates import find_date_phrases

    assert find_date_phrases(message) == phrases


@pytest.mark.parametrize(
    "message, expected",
    [
        ("Please book annual leave from 21 to 23 September.", ("21", "23 September")),
        ("Who in my team is on leave between 5 and 9 October?", ("5", "9 October")),
        ("annual leave 21-23 Sept", ("21", "23 Sept")),
        ("from 28 December 2026 to 4 January 2027", ("28 December 2026", "4 January 2027")),
        ("Book annual leave on 21 September.", None),
        ("Make it until Friday instead", None),
    ],
)
def test_find_date_range(message, expected):
    from app.orchestration.dates import find_date_range

    assert find_date_range(message) == expected


@pytest.mark.parametrize(
    "message, start, end",
    [
        ("Is anyone in my team off next month?", date(2026, 10, 1), date(2026, 10, 31)),
        ("Who is on leave this month?", date(2026, 9, 1), date(2026, 9, 30)),
        ("Team leave in December?", date(2026, 12, 1), date(2026, 12, 31)),
        ("Who is off during feb 2027?", date(2027, 2, 1), date(2027, 2, 28)),
    ],
)
def test_month_period(message, start, end):
    period = find_month_period(message, TODAY)
    assert (period.start, period.end) == (start, end)


@pytest.mark.parametrize("message", ["Who may be off next week?", "Who is off on 5 October?", "What is my balance?"])
def test_no_month_period(message):
    assert find_month_period(message, TODAY) is None


def test_a_weekday_with_its_year_is_that_weekday():
    """Eval finding: the planner turned "until Friday" into "Friday 2026"."""
    assert resolve_date("Friday 2026", TODAY) == date(2026, 9, 18)
    assert resolve_date("next Friday, 2026", TODAY) == date(2026, 9, 18)
    assert resolve_end("Friday 2026", date(2026, 9, 21), TODAY) == date(2026, 9, 25)
    with pytest.raises(DateError):
        resolve_date("Friday 2027", TODAY)  # the coming Friday is not in 2027
