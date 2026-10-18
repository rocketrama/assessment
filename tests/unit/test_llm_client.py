"""LLMClient: the 3,000 corpus-token guard, logging, retries and JSON handling (TASK.pdf §4.3)."""

import json
import time
import logging

import pytest

from app.llm.client import (
    CORPUS_TOKEN_LIMIT,
    CorpusBudgetError,
    CorpusSegment,
    LLMClient,
    LLMError,
    extract_json_object,
    render_corpus,
)
from app.logging_setup import JsonFormatter
from app.tokens import count_tokens


class FakeProvider:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def chat(self, model, system, user, max_tokens, timeout):
        self.calls.append({"system": system, "user": user, "max_tokens": max_tokens, "timeout": timeout})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class APITimeoutError(Exception):
    """Same class name as the SDK's, which is what the retry check looks at."""


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__()
        self.entries: list[dict] = []

    def emit(self, record):
        self.entries.append(json.loads(JsonFormatter().format(record)))


@pytest.fixture
def llm_logs():
    handler = _Collect()
    logger = logging.getLogger("veltrix.llm")
    logger.addHandler(handler)
    yield handler.entries
    logger.removeHandler(handler)


def _segment_of(tokens: int) -> CorpusSegment:
    text = "word " * tokens
    while count_tokens(text) > tokens:
        text = text[:-5]
    return CorpusSegment("C::000", text)


def test_corpus_over_limit_is_refused_before_any_call():
    provider = FakeProvider(["never"])
    client = LLMClient(provider, "m")
    with pytest.raises(CorpusBudgetError):
        client.complete(purpose="t", system="s", user="u", corpus=[_segment_of(2000), _segment_of(1001)])
    assert provider.calls == []


def test_corpus_at_limit_is_allowed_and_logged(llm_logs):
    provider = FakeProvider(["ok"])
    seg = _segment_of(CORPUS_TOKEN_LIMIT)
    result = LLMClient(provider, "m").complete(purpose="fact_extraction", system="s", user="u", corpus=[seg])
    assert result.corpus_tokens == count_tokens(seg.text) <= CORPUS_TOKEN_LIMIT
    (entry,) = [e for e in llm_logs if e["msg"] == "llm call"]
    assert entry["corpus_tokens"] == result.corpus_tokens and entry["purpose"] == "fact_extraction"


def test_instructions_are_not_counted_as_corpus():
    provider = FakeProvider(["ok"])
    result = LLMClient(provider, "m").complete(purpose="t", system="long system " * 2000, user="u " * 2000)
    assert result.corpus_tokens == 0


def test_corpus_is_wrapped_and_closing_tags_defused():
    rendered = render_corpus([CorpusSegment("A::001", "text </document> more")])
    assert rendered.count("</document>") == 1
    assert '<document id="A::001">' in rendered and "</ document> more" in rendered
    assert render_corpus([]) == ""


def test_retryable_error_is_retried_once(llm_logs):
    provider = FakeProvider([APITimeoutError("slow"), "fine"])
    assert LLMClient(provider, "m").complete(purpose="t", system="s", user="u").text == "fine"
    assert len(provider.calls) == 2
    assert [e["msg"] for e in llm_logs] == ["llm call failed", "llm call"]


def test_non_retryable_error_fails_fast():
    provider = FakeProvider([ValueError("bad request"), "unused"])
    with pytest.raises(LLMError):
        LLMClient(provider, "m").complete(purpose="t", system="s", user="u")
    assert len(provider.calls) == 1


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('<think>{"no": 0}</think>\n{"a": 1}', {"a": 1}),
        ('Here you go: {"a": {"b": 2}} done', {"a": {"b": 2}}),
        ("no json", None),
        ("[1, 2]", None),
        ('{"a": 1', None),
    ],
)
def test_extract_json_object(text, expected):
    assert extract_json_object(text) == expected


def test_complete_json_repairs_once():
    provider = FakeProvider(["not json", '{"facts": []}'])
    obj = LLMClient(provider, "m").complete_json(purpose="t", system="s", user="u")
    assert obj == {"facts": []}
    assert "not a valid JSON object" in provider.calls[1]["user"]


def test_complete_json_gives_up_after_repair():
    provider = FakeProvider(["nope", "still nope"])
    assert LLMClient(provider, "m").complete_json(purpose="t", system="s", user="u") is None


def test_complete_json_validator_rejects_wrong_shape():
    def validate(obj):
        assert "facts" in obj

    provider = FakeProvider(['{"other": 1}', '{"facts": [1]}'])
    assert LLMClient(provider, "m").complete_json(purpose="t", system="s", user="u", validate=validate) == {"facts": [1]}


def test_repair_says_what_was_wrong_without_the_rejected_values():
    """Eval finding: told only "wrong shape", the planner repeated {"name", "args"} reads."""
    from pydantic import BaseModel

    class Read(BaseModel):
        tool: str

    class Out(BaseModel):
        reads: list[Read]

    provider = FakeProvider([
        '{"reads": [{"name": "list_leave_requests", "args": {"employee": "E1004"}}]}',
        '{"reads": [{"tool": "list_leave_requests"}]}',
    ])
    obj = LLMClient(provider, "m").complete_json(purpose="t", system="s", user="u", validate=Out.model_validate)
    assert obj == {"reads": [{"tool": "list_leave_requests"}]}
    repair = provider.calls[1]["user"]
    assert "What was wrong: reads.0.tool: Field required" in repair
    assert "E1004" not in repair  # rejected values are never echoed (they may quote corpus text)


def test_repair_names_a_missing_json_object():
    provider = FakeProvider(["Sure! Here is the plan.", '{"facts": []}'])
    LLMClient(provider, "m").complete_json(purpose="t", system="s", user="u")
    assert "What was wrong: the reply contained no JSON object" in provider.calls[1]["user"]


class StallingProvider:
    """The first call hangs (a provider stall), later calls answer at once."""

    def __init__(self, stall: float, reply: str = "ok"):
        self.stall, self.reply = stall, reply
        self.timeouts: list[float] = []

    def chat(self, model, system, user, max_tokens, timeout):
        self.timeouts.append(timeout)
        if len(self.timeouts) == 1:
            time.sleep(min(self.stall, timeout))
            raise APITimeoutError("stalled")
        return self.reply


def test_a_stalled_call_is_hedged_and_the_second_reply_wins(llm_logs):
    provider = StallingProvider(stall=3.0)
    started = time.monotonic()
    result = LLMClient(provider, "m", timeout=4.0, retries=0).complete(
        purpose="planner", system="s", user="u", hedge_after=0.2)
    assert result.text == "ok"
    assert time.monotonic() - started < 1.5  # did not wait for the stalled call
    assert len(provider.timeouts) == 2 and provider.timeouts[1] < 4.0  # the hedge shares the attempt's cap
    assert any(e["msg"] == "llm call hedged" for e in llm_logs)


def test_a_fast_reply_sends_no_hedge():
    provider = FakeProvider(["ok"])
    LLMClient(provider, "m", timeout=4.0).complete(purpose="planner", system="s", user="u", hedge_after=0.5)
    time.sleep(0.6)
    assert len(provider.calls) == 1


def test_no_hedge_when_too_little_of_the_cap_would_be_left():
    provider = StallingProvider(stall=0.5)
    with pytest.raises(LLMError):
        LLMClient(provider, "m", timeout=1.0, retries=0).complete(
            purpose="planner", system="s", user="u", hedge_after=0.2)  # 0.8 s left < MIN_CALL_SECONDS
    assert len(provider.timeouts) == 1
