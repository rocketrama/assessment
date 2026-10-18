"""The per-request deadline caps every LLM attempt (§9.1: 30 s per /chat)."""

import time

import pytest

from app.llm.client import HARD_CAP_GRACE_SECONDS, MIN_CALL_SECONDS, DeadlineExceeded, LLMClient, LLMError


class APITimeoutError(Exception):
    """Same class name as the SDK's, which is what makes it retryable."""


class Recorder:
    def __init__(self, replies, delay=0.0):
        self.replies, self.timeouts, self.delay = list(replies), [], delay

    def chat(self, model, system, user, max_tokens, timeout):
        self.timeouts.append(timeout)
        time.sleep(self.delay)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_a_provider_that_ignores_its_timeout_is_cut_off():
    """Wall-clock cap: the SDK timeout alone let a /chat reach 35.1 s (eval round 6)."""

    provider = Recorder(["too late"], delay=3.0)  # hangs well past its 1 s timeout
    started = time.monotonic()
    with pytest.raises(LLMError):
        LLMClient(provider, "m", timeout=1, retries=0).complete(purpose="t", system="s", user="u")
    assert time.monotonic() - started < 1 + HARD_CAP_GRACE_SECONDS + 0.5


def test_a_hung_call_never_overruns_the_request_deadline():
    provider = Recorder(["late", "late"], delay=5.0)
    started = time.monotonic()
    with pytest.raises(Exception):
        LLMClient(provider, "m", timeout=12, retries=1).complete(
            purpose="t", system="s", user="u", deadline=time.monotonic() + 2.5
        )
    assert time.monotonic() - started < 2.5 + HARD_CAP_GRACE_SECONDS + 0.5


def test_attempt_gets_only_the_time_left():
    provider = Recorder(["ok"])
    LLMClient(provider, "m", timeout=12).complete(purpose="t", system="s", user="u", deadline=time.monotonic() + 5)
    assert 4 < provider.timeouts[0] <= 5


def test_per_call_timeout_still_applies_when_more_time_is_left():
    provider = Recorder(["ok"])
    LLMClient(provider, "m", timeout=12).complete(purpose="t", system="s", user="u", deadline=time.monotonic() + 60)
    assert provider.timeouts == [12]


def test_no_call_when_too_little_time_is_left():
    provider = Recorder(["never"])
    with pytest.raises(DeadlineExceeded):
        LLMClient(provider, "m").complete(purpose="t", system="s", user="u", deadline=time.monotonic() + MIN_CALL_SECONDS / 2)
    assert provider.timeouts == []


def test_no_retry_when_the_deadline_is_near():
    provider = Recorder([APITimeoutError("slow"), "unused"])
    with pytest.raises(DeadlineExceeded):
        LLMClient(provider, "m", retries=1).complete(
            purpose="t", system="s", user="u", deadline=time.monotonic() + MIN_CALL_SECONDS + 0.5
        )
    assert len(provider.timeouts) == 1  # the retry (after a 1 s back-off) would not fit


def test_complete_json_skips_the_repair_call_without_time():
    provider = Recorder(["not json"], delay=0.2)  # the first call uses up the margin
    with pytest.raises(DeadlineExceeded):
        LLMClient(provider, "m", retries=0).complete_json(
            purpose="t", system="s", user="u", deadline=time.monotonic() + MIN_CALL_SECONDS + 0.1
        )
    assert len(provider.timeouts) == 1
