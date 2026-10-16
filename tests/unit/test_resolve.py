"""precedence.resolve: one rule across documents — winner, conflict, upcoming (TASK.pdf §5)."""

from datetime import date

from app.retrieval.corpus import FactRecord
from app.retrieval.eligibility import DocMeta
from app.retrieval.precedence import resolve, value_key

D = date.fromisoformat
TODAY = D("2026-09-15")

# TASK.pdf §5.2 worked example (invented by the spec), as facts of one rule.
DOCS = {
    "V1": DocMeta("V1", "policy", ("GLOBAL",), D("2025-01-01"), "all"),
    "V2": DocMeta("V2", "policy", ("GLOBAL",), D("2026-04-01"), "all"),
    "FAQ": DocMeta("FAQ", "faq", ("GLOBAL",), D("2026-05-01"), "all"),
    "DE": DocMeta("DE", "addendum", ("DE",), D("2025-06-01"), "all"),
    "V3": DocMeta("V3", "policy", ("GLOBAL",), D("2027-01-01"), "all"),
    "MGR": DocMeta("MGR", "handbook", ("GLOBAL",), D("2026-01-01"), "managers"),
}


def fact(doc_id, value):
    return FactRecord(f"{doc_id}::000#f00", f"{doc_id}::000", doc_id, "volunteer-days", "volunteer", "days", value, "days", None)


EXAMPLE = [fact("V1", "2"), fact("V2", "3"), fact("FAQ", "2"), fact("DE", "5"), fact("V3", "4")]


def run(region, reference=TODAY, role="employee", facts=EXAMPLE, include_upcoming=True):
    return resolve("volunteer-days", facts, DOCS, role, region, reference, include_upcoming)


def values(facts):
    return [(f.doc_id, f.value) for f in facts]


def test_india_today_v2_wins_and_v3_is_upcoming():
    r = run("IN")
    assert values(r.winners) == [("V2", "3")]
    assert sorted(values(r.outranked)) == [("FAQ", "2"), ("V1", "2")]
    assert values(r.upcoming) == [("V3", "4")]
    assert not r.conflict


def test_germany_today_addendum_wins_and_v3_is_not_mentioned():
    r = run("DE")
    assert values(r.winners) == [("DE", "5")]
    assert r.upcoming == ()  # the addendum would still win on scope in 2027


def test_india_past_date_only_v1_and_no_upcoming():
    r = run("IN", reference=D("2025-08-01"), include_upcoming=False)
    assert values(r.winners) == [("V1", "2")]
    assert r.upcoming == () and r.outranked == ()


def test_full_tie_with_different_values_is_a_conflict():
    docs_tie = [fact("V2", "3"), FactRecord("X::000#f00", "X::000", "X", "volunteer-days", "v", "d", "6", None, None)]
    DOCS["X"] = DocMeta("X", "policy", ("GLOBAL",), D("2026-04-01"), "all")
    try:
        r = run("IN", facts=docs_tie)
        assert r.conflict and r.winning_doc_ids == ("V2", "X")
    finally:
        del DOCS["X"]


def test_unreadable_document_never_wins_or_appears():
    facts = EXAMPLE + [fact("MGR", "9")]
    assert all(f.doc_id != "MGR" for f in run("IN").winners + run("IN").outranked)
    assert values(run("IN", role="manager", facts=facts).winners) == [("V2", "3")]  # policy beats handbook on recency


def test_rule_with_only_a_future_document_is_upcoming_only():
    r = run("IN", facts=[fact("V3", "4")])
    assert r.winners == () and values(r.upcoming) == [("V3", "4")]
    assert run("IN", facts=[fact("V3", "4")], include_upcoming=False) is None


def test_other_region_only_gives_nothing():
    assert run("IN", facts=[fact("DE", "5")]) is None


def test_value_key_treats_number_words_and_units_alike():
    assert value_key("ten working days") == value_key("10") == value_key("10 days")
    assert value_key("12") != value_key("10")
    assert value_key("not paid out") == value_key("Not  paid out")
