"""The provider adapters send what each SDK expects (§10.3: reviewers use their own
OpenAI / Anthropic key). The SDK clients are replaced by recorders — no network."""

from types import SimpleNamespace

import pytest

from app.config import Settings
from app.llm.client import AnthropicProvider, LLMClient, OpenAICompatibleProvider


class FakeCompletions:
    def __init__(self, reply):
        self.reply, self.kwargs = reply, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.reply


def openai_with(reply_text, reasoning_effort=""):
    provider = OpenAICompatibleProvider("sk-test", None, reasoning_effort)
    completions = FakeCompletions(SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply_text))]))
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return provider, completions


def test_openai_request_shape():
    provider, completions = openai_with('{"ok": 1}')
    assert provider.chat("gpt-x", "SYSTEM", "USER", 500, 9.5) == '{"ok": 1}'
    kw = completions.kwargs
    assert kw["model"] == "gpt-x" and kw["temperature"] == 0 and kw["max_tokens"] == 500 and kw["timeout"] == 9.5
    assert kw["messages"] == [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "USER"}]
    assert kw["extra_body"] == {}  # nothing provider-specific unless configured


def test_reasoning_effort_only_when_configured():
    provider, completions = openai_with("x", reasoning_effort="low")
    provider.chat("m", "s", "u", 10, 1)
    assert completions.kwargs["extra_body"] == {"reasoning_effort": "low"}


def test_openai_empty_content_is_empty_string():
    provider, _ = openai_with(None)
    assert provider.chat("m", "s", "u", 10, 1) == ""


def test_anthropic_request_shape():
    provider = AnthropicProvider("sk-ant-test")
    messages = FakeCompletions(SimpleNamespace(content=[
        SimpleNamespace(type="thinking", text="ignored"),
        SimpleNamespace(type="text", text='{"a": '),
        SimpleNamespace(type="text", text="1}"),
    ]))
    provider._client = SimpleNamespace(messages=messages)
    assert provider.chat("claude-x", "SYSTEM", "USER", 700, 8.0) == '{"a": 1}'
    kw = messages.kwargs
    assert kw["system"] == "SYSTEM" and kw["messages"] == [{"role": "user", "content": "USER"}]
    assert kw["temperature"] == 0 and kw["max_tokens"] == 700 and kw["timeout"] == 8.0


@pytest.mark.parametrize(
    "provider, key_field, expected",
    [
        ("openai", "openai_api_key", OpenAICompatibleProvider),
        ("nvidia", "nvidia_api_key", OpenAICompatibleProvider),
        ("anthropic", "anthropic_api_key", AnthropicProvider),
    ],
)
def test_from_settings_picks_the_provider(provider, key_field, expected):
    settings = Settings(_env_file=None, llm_provider=provider, llm_model="m", **{key_field: "k"})
    client = LLMClient.from_settings(settings)
    assert isinstance(client.provider, expected) and client.model == "m"


@pytest.mark.parametrize(
    "fields",
    [
        {"llm_provider": "openai", "llm_model": ""},  # no model
        {"llm_provider": "openai", "llm_model": "m"},  # no key
        {"llm_provider": "anthropic", "llm_model": "m"},  # no key
    ],
)
def test_from_settings_without_model_or_key_disables_the_llm(fields):
    assert LLMClient.from_settings(Settings(_env_file=None, **fields)) is None
