"""Section 5 eligibility and precedence, including the spec's worked example (TASK.pdf §5.2)."""

from datetime import date

import pytest

from app.retrieval.eligibility import DocMeta, can_read, is_eligible
from app.retrieval.precedence import winners

D = date.fromisoformat

# TASK.pdf §5.2 — invented by the spec, not from the corpus.
V1 = DocMeta("v1", "policy", ("GLOBAL",), D("2025-01-01"), "all")
V2 = DocMeta("v2", "policy", ("GLOBAL",), D("2026-04-01"), "all")
FAQ = DocMeta("faq", "faq", ("GLOBAL",), D("2026-05-01"), "all")
DE_ADD = DocMeta("de", "addendum", ("DE",), D("2025-06-01"), "all")
V3 = DocMeta("v3", "policy", ("GLOBAL",), D("2027-01-01"), "all")
EXAMPLE = [V1, V2, FAQ, DE_ADD, V3]
TODAY = D("2026-09-15")


def resolve(region, reference_date, role="employee"):
    return [d.doc_id for d in winners([d for d in EXAMPLE if is_eligible(d, role, region, reference_date)])]


def test_worked_example_india_today():
    assert resolve("IN", TODAY) == ["v2"]  # FAQ loses on authority, v2 beats v1 on recency


def test_worked_example_india_upcoming_v3_would_win_on_its_date():
    assert resolve("IN", V3.effective_date) == ["v3"]


def test_worked_example_germany_today_and_in_2027():
    assert resolve("DE", TODAY) == ["de"]
    assert resolve("DE", V3.effective_date) == ["de"]  # addendum still wins on scope → v3 not mentioned


def test_worked_example_india_past_date():
    assert resolve("IN", D("2025-08-01")) == ["v1"]


def test_full_tie_is_a_conflict():
    a = DocMeta("a", "policy", ("IN",), D("2026-01-01"), "all")
    b = DocMeta("b", "addendum", ("IN", "US"), D("2026-01-01"), "all")
    assert [d.doc_id for d in winners([a, b])] == ["a", "b"]


def test_wiki_loses_to_older_policy():
    wiki = DocMeta("wiki", "wiki", ("IN",), D("2026-06-01"), "all")
    assert [d.doc_id for d in winners([V1, wiki])] == ["v1"]  # authority before scope and recency


@pytest.mark.parametrize(
    "role, readable",
    [
        ("employee", {"all"}),
        ("manager", {"all", "managers"}),
        ("hr", {"all", "managers", "hr"}),
        ("contractor", set()),
    ],
)
def test_access_levels_per_role(role, readable):
    assert {a for a in ("all", "managers", "hr") if can_read(role, a)} == readable


def test_eligibility_checks_all_three_conditions():
    doc = DocMeta("m", "handbook", ("US",), D("2026-01-01"), "managers")
    assert is_eligible(doc, "manager", "US", TODAY)
    assert not is_eligible(doc, "employee", "US", TODAY)  # access
    assert not is_eligible(doc, "manager", "IN", TODAY)  # region
    assert not is_eligible(doc, "manager", "US", D("2025-12-31"))  # not yet effective
