"""Validator: grounded citations, route invariants (§8.1) and answer scrubbing (§7.4)."""

import pytest

from app.api.schemas import Citation, ToolCall
from app.orchestration.validator import (
    finalize,
    grounded_chunk_ids,
    route_for,
    scrub_answer,
    strip_internal_ids,
    superseded_values,
)
from app.retrieval.corpus import FactRecord
from app.retrieval.precedence import RuleResolution


def F(chunk_id, value):
    return FactRecord(f"{chunk_id}#f00", chunk_id, chunk_id.split("::")[0], "carry", "carry forward", "maximum", value, "days", None)


# Germany employee: the DE addendum wins; global v2 (same value) is outranked.
CARRY = RuleResolution("carry", winners=(F("ADD-DE::001", "10"),), outranked=(F("POL-V2::006", "10"),), upcoming=())
DEADLINE = RuleResolution("deadline", winners=(F("POL-V2::006", "31 December"),), outranked=(), upcoming=())


def test_outranked_only_source_is_not_cited():
    used = ["ADD-DE::001", "POL-V2::006"]
    answer = "You may carry forward up to 10 days."
    assert grounded_chunk_ids(used, answer, [CARRY, DEADLINE]) == ["ADD-DE::001"]


def test_source_is_kept_when_its_winning_value_is_in_the_answer():
    answer = "Up to 10 days; anything above the limit is forfeited on 31 December."
    assert grounded_chunk_ids(["ADD-DE::001", "POL-V2::006"], answer, [CARRY, DEADLINE]) == ["ADD-DE::001", "POL-V2::006"]


def test_chunks_without_facts_are_trusted():
    assert grounded_chunk_ids(["CAL::001"], "Deepavali is on 9 November.", [CARRY]) == ["CAL::001"]


def test_winning_source_is_kept_even_for_a_statement_not_extracted_as_a_fact():
    """Eval finding: the answer relied on "carried forward days do not expire", which
    the extraction missed; the addendum's recorded facts (15 days) aren't in the answer."""
    addendum = RuleResolution("carry-in", winners=(F("ADD-IN::001", "15"),), outranked=(F("POL-V2::006", "10"),), upcoming=())
    answer = "Carried-forward days in India do not expire on a fixed date."
    assert grounded_chunk_ids(["ADD-IN::001", "POL-V2::006"], answer, [addendum]) == ["ADD-IN::001"]


def test_number_words_match_digits():
    # V2::005 also holds an outranked fact, so its winning value must be in the answer.
    r = RuleResolution("min", winners=(F("V2::005", "ten working days"),), outranked=(), upcoming=())
    lost = RuleResolution("other", winners=(F("ADD::001", "7"),), outranked=(F("V2::005", "9"),), upcoming=())
    assert grounded_chunk_ids(["V2::005"], "Take at least 10 working days.", [r, lost]) == ["V2::005"]
    assert grounded_chunk_ids(["V2::005"], "Take at least 12 working days.", [r, lost]) == []


def test_faq_citation_dropped_when_its_value_is_not_in_the_answer():
    """Eval finding: the FAQ's outdated "3 days" formed its own rule and was cited next to the policy's 5."""
    r = RuleResolution("bereavement", winners=(F("POL-V2::009", "5"),), outranked=(), upcoming=())
    faq = {"FAQ::005": ["3"]}
    assert grounded_chunk_ids(["POL-V2::009", "FAQ::005"], "Up to 5 working days.", [r], faq) == ["POL-V2::009"]
    # agreeing with the policy: kept
    assert grounded_chunk_ids(["POL-V2::009", "FAQ::005"], "5 days; the FAQ's 3 is older.", [r], faq) == ["POL-V2::009", "FAQ::005"]


def test_faq_alone_is_kept():
    """The only source for a point (e.g. an FAQ answer the policy doesn't restate in the passages)."""
    assert grounded_chunk_ids(["FAQ::008"], "No, leave is taken in full days.", [], {"FAQ::008": ["whole days"]}) == ["FAQ::008"]
    assert grounded_chunk_ids(["FAQ::008", "POL::001"], "No, half days are not possible.", [], {"FAQ::008": []}) == ["FAQ::008", "POL::001"]


def test_rule_table_alone_is_cited_through_its_winning_passage():
    """Eval finding: "a PIP lasts 60 calendar days" taken from the rule table alone lost
    every citation and was replaced by "the documents don't cover this"."""
    pip = RuleResolution("pip-duration", winners=(F("HR-PIP::001", "60"),), outranked=(), upcoming=())
    assert grounded_chunk_ids(["rule-table"], "A PIP lasts 60 calendar days.", [pip]) == ["HR-PIP::001"]


@pytest.mark.parametrize("named", ["rule pip-duration", "pip-duration"])
def test_a_rule_row_named_as_the_source_is_cited_through_its_passage(named):
    """Live finding: the composer named "rule performance-improvement-plan-duration"."""
    pip = RuleResolution("pip-duration", winners=(F("HR-PIP::001", "60"),), outranked=(), upcoming=())
    ext = RuleResolution("pip-extension", winners=(F("HR-PIP::002", "30"),), outranked=(), upcoming=())
    assert grounded_chunk_ids([named], "A PIP lasts 60 days, extendable by 30.", [pip, ext]) == ["HR-PIP::001"]


def test_rule_table_never_cites_an_outranked_passage():
    # Germany: DE addendum wins with 10; global v2 holds the outranked 10 too.
    assert grounded_chunk_ids(["rule-table"], "You can carry forward 10 days.", [CARRY]) == ["ADD-DE::001"]


def test_rule_table_cites_nothing_when_no_value_is_in_the_answer():
    assert grounded_chunk_ids(["rule-table"], "Please ask your manager.", [CARRY, DEADLINE]) == []


def test_rule_table_and_a_passage_are_not_cited_twice():
    used = ["rule-table", "ADD-DE::001"]
    assert grounded_chunk_ids(used, "You can carry forward 10 days.", [CARRY]) == ["ADD-DE::001"]


def test_upcoming_value_supports_a_citation():
    r = RuleResolution("sick", winners=(F("V2::007", "12"),), outranked=(), upcoming=(F("V3::001", "14"),))
    assert grounded_chunk_ids(["V2::007", "V3::001"], "12 days now; 14 from 2027.", [r]) == ["V2::007", "V3::001"]


CITE = [Citation(doc_id="D", locator="L")]
CALL = [ToolCall(name="get_leave_balance", args={"employee_id": "E1001"}, status="success")]


@pytest.mark.parametrize(
    "citations, calls, route", [(CITE, [], "rag"), ([], CALL, "tool"), (CITE, CALL, "hybrid"), ([], [], "no_answer")]
)
def test_route_from_fields(citations, calls, route):
    assert route_for(citations, calls) == route


@pytest.mark.parametrize("route", ["clarify", "no_answer", "refuse"])
def test_empty_routes_clear_fields(route):
    r = finalize("id", "x", route, CITE, CALL, pending_confirmation=False)
    assert (r.citations, r.tool_calls, r.pending_confirmation) == ([], [], False)


def test_mismatched_route_is_an_error():
    with pytest.raises(ValueError):
        finalize("id", "x", "hybrid", CITE, [])


def test_pending_needs_a_pending_tool_call():
    with pytest.raises(ValueError):
        finalize("id", "x", "tool", [], CALL, pending_confirmation=True)


def test_scrub_answer():
    text = "Ask Priya Nair (p.nair@veltrix.example, +91 98765 43210) or Sam."
    out = scrub_answer(text, hidden_names=["Priya Nair"])
    assert out == "Ask [name removed] (p***@veltrix.example, [phone removed]) or Sam."


# --- answer text checks -------------------------------------------------------

def Fv(chunk_id, value, unit="days"):
    return FactRecord(f"{chunk_id}#f00", chunk_id, chunk_id.split("::")[0], "r", "t", "a", value, unit, None)


# India: the addendum wins with "do not expire"; v1's 30 June and v2's 31 March lost.
EXPIRY = RuleResolution(
    "carry-deadline",
    winners=(Fv("ADD-IN::001", "do not expire on a fixed date", None),),
    outranked=(Fv("POL-V1::004", "30 June", None), Fv("POL-V2::006", "31 March", None)),
    upcoming=(),
)


def test_superseded_values_in_the_answer_are_found():
    """Eval finding: the correct answer also said "other versions set 30 June or 31 March"."""
    answer = "There is no fixed deadline. Other versions set deadlines of 30 June or 31 March."
    assert superseded_values(answer, [EXPIRY]) == ["30 june", "31 march"]
    assert superseded_values("Carried-forward days do not expire on a fixed date.", [EXPIRY]) == []


def test_a_value_that_is_also_current_is_not_superseded():
    # Germany: the DE addendum's 10 days wins; v2's 10 days lost on precedence, not on value.
    assert superseded_values("You can carry forward 10 days.", [CARRY]) == []
    other = RuleResolution("other", winners=(Fv("X::001", "31 March", None),), outranked=(), upcoming=())
    assert superseded_values("Use them by 31 March.", [EXPIRY, other]) == []


def test_bare_numbers_need_their_unit():
    """"10" alone also matches times and balances; it counts only as "10 days"."""
    lost = RuleResolution("sick", winners=(Fv("V2::001", "12"),), outranked=(Fv("V1::001", "10"),), upcoming=())
    assert superseded_values("You get 12 days. Core hours start at 10:00; your balance is 10.", [lost]) == []
    assert superseded_values("You get 12 days (it used to be 10 days).", [lost]) == ["10 days"]


@pytest.mark.parametrize(
    "answer, expected",
    [
        # the eval answer, with U+2011 non-breaking hyphens
        ("The India addendum (ADD‑LEAVE‑IN::001) states that days do not expire.",
         "The India addendum states that days do not expire."),
        ("See ADD-LEAVE-IN::001 and POL-LEAVE-GLOBAL-V2 for details.", "See and for details."),
        ("Days do not expire [ADD-LEAVE-IN::001, POL-LEAVE-GLOBAL-V2::006].", "Days do not expire."),
        ("Line one.\nLine two.", "Line one.\nLine two."),
    ],
)
def test_internal_ids_are_removed(answer, expected):
    assert strip_internal_ids(answer, ["ADD-LEAVE-IN", "POL-LEAVE-GLOBAL-V2", "POL-LEAVE-GLOBAL-V2-SUMMARY"]) == expected


def test_words_that_only_look_like_ids_are_kept():
    assert strip_internal_ids("Policy covers IN and US offices.", ["ADD-LEAVE-IN"]) == "Policy covers IN and US offices."
