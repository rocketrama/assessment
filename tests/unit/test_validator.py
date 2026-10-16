"""Validator: grounded citations, route invariants (§8.1) and answer scrubbing (§7.4)."""

import pytest

from app.api.schemas import Citation, ToolCall
from app.orchestration.validator import finalize, grounded_chunk_ids, route_for, scrub_answer
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
