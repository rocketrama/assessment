"""LLM "same rule?" judge: verdicts, caching, failures and the deadline. The LLM is faked;
replies are built from the question ids it is sent."""

import json
import re
import sqlite3
import time

import pytest

from app.ingestion.extract_facts import Fact
from app.ingestion.rule_judge import SYSTEM_PROMPT, JudgeStats, _AnswersOut, judge_pairs, parse_verdicts
from app.llm.client import LLMClient


def F(fact_id, topic, attribute, value, quote):
    return Fact(fact_id, fact_id.split("#")[0], fact_id.split("::")[0], topic, attribute, value, "days", None, quote)


POLICY = F("POL::001#f00", "remote work days", "maximum days per week", "2", "up to 2 days per week")
MEMO = F("MEMO::000#f00", "hybrid work", "days per week", "3", "may work from home for up to 3 days per week")
LEARN = F("LRN::001#f00", "learning days", "days per year", "3", "3 learning days per year")
WELL = F("WB::001#f00", "wellbeing days", "days per year", "2", "2 wellbeing days per year")


class ScriptedProvider:
    """Answers "same" for questions whose text contains a phrase in `same_if`."""

    def __init__(self, same_if=("hybrid work",), fail=False, delay=0.0):
        self.same_if, self.fail, self.delay = same_if, fail, delay
        self.requests: list[str] = []

    def chat(self, model, system, user, max_tokens, timeout):
        self.requests.append(user)
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("provider down")
        blocks = re.findall(r'<document id="(q\d+)">\n(.*?)\n</document>', user, re.DOTALL)
        answers = {qid: any(p in text for p in self.same_if) for qid, text in blocks}
        return json.dumps({"answers": answers})


def run(questions, provider, conn=None, deadline=None):
    conn = conn or sqlite3.connect(":memory:", isolation_level=None)
    stats = JudgeStats()
    verdicts = judge_pairs(
        questions,
        LLMClient(provider, "m", retries=0),
        conn,
        concurrency=2,
        deadline=deadline or time.monotonic() + 30,
        stats=stats,
    )
    return verdicts, stats, conn


def test_verdicts_follow_the_llm_in_question_order():
    verdicts, stats, _ = run([([POLICY], [MEMO]), ([LEARN], [WELL])], ScriptedProvider())
    assert verdicts == [True, False]
    assert stats.same == 1 and stats.unanswered == 0


def test_quotes_travel_as_corpus_not_instructions():
    provider = ScriptedProvider()
    run([([POLICY], [MEMO])], provider)
    instructions, corpus = provider.requests[0].split("<documents>", 1)
    assert "may work from home" not in instructions and "may work from home" in corpus


def test_answers_are_cached_so_reingestion_asks_nothing():
    provider = ScriptedProvider()
    _, first, conn = run([([POLICY], [MEMO]), ([LEARN], [WELL])], provider)
    verdicts, again, _ = run([([POLICY], [MEMO]), ([LEARN], [WELL])], provider, conn)
    assert first.llm_calls == 1
    assert verdicts == [True, False]
    assert again.llm_calls == 0 and again.cache_hits == 2


def test_failed_call_means_no_merge_and_is_not_cached():
    _, _, conn = run([([POLICY], [MEMO])], ScriptedProvider(fail=True))
    verdicts, stats, _ = run([([POLICY], [MEMO])], ScriptedProvider(fail=True), conn)
    assert verdicts == [None] and stats.unanswered == 1 and stats.cache_hits == 0
    verdicts, _, _ = run([([POLICY], [MEMO])], ScriptedProvider(), conn)  # asked again once the LLM is back
    assert verdicts == [True]


def test_deadline_leaves_pairs_unanswered():
    verdicts, stats, _ = run([([POLICY], [MEMO])], ScriptedProvider(delay=0.5), deadline=time.monotonic() + 0.05)
    assert verdicts == [None] and stats.unanswered == 1


def test_no_questions_no_call():
    provider = ScriptedProvider()
    verdicts, stats, _ = run([], provider)
    assert verdicts == [] and not provider.requests and stats.llm_calls == 0


def test_place_only_conditions_are_not_shown_as_a_subgroup():
    """Eval finding: "conditions: Bengaluru and Austin offices" made the model keep the
    WFH memo apart from the policy it replaces. Real subgroups are still shown."""
    from dataclasses import replace

    from app.ingestion.rule_judge import question_text

    memo = replace(MEMO, conditions="Bengaluru and Austin offices")
    carer = replace(POLICY, conditions="primary caregiver")
    text = question_text([memo], [carer])
    assert "Bengaluru" not in text.split("quote")[0]
    assert "conditions: primary caregiver" in text
    # a unit already in the value is not repeated ("3 days days")
    assert "= 3 days (quote" in question_text([replace(MEMO, value="3 days")], [POLICY])


def test_the_reply_is_a_flat_id_to_verdict_map():
    """Eval finding: as a list of {"id", "same"} objects the model kept writing
    `{"id":"q38","same":false]}` (no closing brace) and those batches were never judged."""
    assert '{"answers": {"q1": true, "q2": false}}' in SYSTEM_PROMPT
    assert _AnswersOut.model_validate({"answers": {"q1": True, "q2": False}}).answers == {"q1": True, "q2": False}


@pytest.mark.parametrize("reply", [
    '{"answers":{"q1":false,"q2":true,"q3":false"}}',  # stray quote, seen from gpt-oss-20b
    '{"answers":{"q1":false,"q2":true,"q3":false}',  # missing brace
    '{"answers":[{"id":"q1","same":false}]}  "q1": false, "q2": true, "q3": false',
])
def test_one_slip_in_the_reply_does_not_lose_the_batch(reply):
    assert parse_verdicts(reply) == {"q1": False, "q2": True, "q3": False}


def test_verdicts_only_count_for_questions_asked_and_an_empty_reply_fails():
    class Fake:
        def __init__(self, text):
            self.text = text

        def chat(self, model, system, user, max_tokens, timeout):
            return self.text

    conn = sqlite3.connect(":memory:", isolation_level=None)
    questions = [([POLICY], [MEMO]), ([LEARN], [WELL])]
    verdicts = judge_pairs(questions, LLMClient(Fake('{"q0": true, "q7": true'), "m"), conn,
                           concurrency=1, deadline=time.monotonic() + 30)
    assert verdicts == [True, None]  # q7 was never asked; q1 got no verdict and isn't cached
    stats = JudgeStats()
    assert judge_pairs(questions[1:], LLMClient(Fake("I cannot tell."), "m"), conn,
                       concurrency=1, deadline=time.monotonic() + 30, stats=stats) == [None]
