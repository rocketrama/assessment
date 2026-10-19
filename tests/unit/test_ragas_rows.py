"""eval/ragas_eval.py: which eval turns RAGAS scores, and the 3,000-token cap on judge calls.
RAGAS itself lives in a separate venv (requirements-eval.txt) and is not imported here."""

from pathlib import Path

import yaml

from eval.ragas_eval import build_rows

ROOT = Path(__file__).resolve().parents[2]


def result(run=1, route="rag", reference="12 days.", contexts=None, case="c"):
    contexts = [{"id": "POL::007", "kind": "corpus", "text": "Employees receive 12 working days."}] if contexts is None else contexts
    return {"run": run, "case": case, "turn": 0, "say": "How many sick days?", "reference": reference,
            "body": {"route": route, "answer": "You get 12 days."}, "contexts": contexts}


def test_document_answers_become_ragas_samples():
    tool = {"id": "tool:get_leave_balance", "kind": "tool", "text": '{"balance": 6}'}
    rows, skipped = build_rows([result(contexts=[*result()["contexts"], tool])], {1})
    assert rows == [{
        "run": 1, "case": "c", "turn": 0, "user_input": "How many sick days?", "response": "You get 12 days.",
        "retrieved_contexts": ["Employees receive 12 working days.", '{"balance": 6}'],
        "context_ids": ["POL::007", "tool:get_leave_balance"], "reference": "12 days.",
    }]
    assert not any(skipped.values())


def test_turns_that_cannot_be_scored_are_counted_not_scored():
    results = [result(route="no_answer"), result(route="refuse"), result(reference=None), result(contexts=[]), result(run=2)]
    rows, skipped = build_rows(results, {1})
    assert rows == []
    assert skipped == {"not a document answer": 2, "no reference": 1, "no contexts": 1, "over 3,000 corpus tokens": 0}


def test_a_turn_above_the_corpus_budget_is_skipped_not_truncated():
    big = [{"id": f"D::{i}", "kind": "corpus", "text": "leave policy " * 400} for i in range(4)]  # ~3,200 tokens
    rows, skipped = build_rows([result(contexts=big)], {1})
    assert rows == [] and skipped["over 3,000 corpus tokens"] == 1


def test_tool_results_and_notes_do_not_count_as_corpus_tokens():
    tool = [{"id": "tool:x", "kind": "tool", "text": "balance " * 4000}]
    rows, _ = build_rows([result(contexts=[*result()["contexts"], *tool])], {1})
    assert len(rows) == 1


def test_every_document_case_has_a_reference_answer():
    cases = yaml.safe_load((ROOT / "eval" / "cases.yaml").read_text(encoding="utf-8"))
    missing = [
        c["id"] for c in cases for t in c["turns"]
        if set(t.get("route", [])) & {"rag", "hybrid"} and not t.get("reference")
    ]
    assert missing == []
