"""The one way text LLM calls are made (DESIGN §2, TASK.pdf §4.3).

- Corpus text travels separately from instructions, as `CorpusSegment`s. The
  client counts it with cl100k_base and refuses any call above 3,000 corpus
  tokens (must-pass #4), then wraps it in <documents> delimiters as untrusted data.
- Every call is logged with purpose, model, corpus_tokens and latency.
- Temperature 0; one retry on timeouts, rate limits and server errors.
- `complete_json` extracts a JSON object from the reply (tolerating code fences
  and <think> blocks) and makes one repair call if it is invalid.
"""

import json
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from app.config import Settings
from app.tokens import count_tokens

log = logging.getLogger("veltrix.llm")

CORPUS_TOKEN_LIMIT = 3000
REPAIR_INSTRUCTION = (
    "Your previous reply was not a valid JSON object of the requested shape. "
    "Reply again with only that JSON object and nothing else."
)


class CorpusBudgetError(Exception):
    """A call would carry more than CORPUS_TOKEN_LIMIT tokens of corpus text."""


class LLMError(Exception):
    """The provider failed after retries."""


@dataclass(frozen=True)
class CorpusSegment:
    chunk_id: str
    text: str


@dataclass(frozen=True)
class LLMResult:
    text: str
    corpus_tokens: int
    latency_ms: float


class Provider(Protocol):
    def chat(self, model: str, system: str, user: str, max_tokens: int, timeout: float) -> str: ...


class OpenAICompatibleProvider:
    """OpenAI, or any OpenAI-compatible endpoint such as NVIDIA (LLM_BASE_URL)."""

    def __init__(self, api_key: str, base_url: str | None = None, reasoning_effort: str = "") -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        # gpt-oss on NVIDIA: "low" cut a fact-extraction call from ~54 s to ~8 s.
        self._extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}

    def chat(self, model: str, system: str, user: str, max_tokens: int, timeout: float) -> str:
        response = self._client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=max_tokens,
            timeout=timeout,
            extra_body=self._extra,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return response.choices[0].message.content or ""


class AnthropicProvider:
    def __init__(self, api_key: str) -> None:
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key, max_retries=0)

    def chat(self, model: str, system: str, user: str, max_tokens: int, timeout: float) -> str:
        response = self._client.messages.create(
            model=model,
            temperature=0,
            max_tokens=max_tokens,
            timeout=timeout,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if getattr(block, "type", "") == "text")


def _is_retryable(exc: Exception) -> bool:
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    return (
        name in ("APITimeoutError", "APIConnectionError", "RateLimitError", "InternalServerError", "TimeoutError")
        or (isinstance(status, int) and (status == 429 or status >= 500))
    )


def render_corpus(segments: Sequence[CorpusSegment]) -> str:
    """Corpus as delimited, untrusted data. A closing tag inside the text is defused
    so a document can't end the block early."""
    if not segments:
        return ""
    parts = [
        f'<document id="{s.chunk_id}">\n{s.text.replace("</document", "</ document")}\n</document>'
        for s in segments
    ]
    return "\n\n<documents>\n" + "\n".join(parts) + "\n</documents>"


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = _THINK_RE.sub("", text).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        value = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


class LLMClient:
    def __init__(self, provider: Provider, model: str, timeout: float = 12.0, retries: int = 1) -> None:
        self.provider = provider
        self.model = model
        self.timeout = timeout
        self.retries = retries

    @classmethod
    def from_settings(cls, settings: Settings) -> "LLMClient | None":
        if not settings.llm_model:
            log.warning("LLM disabled: LLM_MODEL is not set")
            return None
        provider_name = settings.llm_provider
        if provider_name == "anthropic":
            if not settings.anthropic_api_key:
                log.warning("LLM disabled: no ANTHROPIC_API_KEY")
                return None
            provider: Provider = AnthropicProvider(settings.anthropic_api_key)
        else:
            key = {"nvidia": settings.nvidia_api_key, "openai": settings.openai_api_key}.get(provider_name)
            if not key:
                log.warning("LLM disabled: no API key for provider", extra={"provider": provider_name})
                return None
            provider = OpenAICompatibleProvider(key, settings.llm_base_url or None, settings.llm_reasoning_effort)
        return cls(provider, settings.llm_model, timeout=settings.llm_timeout_seconds)

    def complete(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        corpus: Sequence[CorpusSegment] = (),
        max_tokens: int = 1024,
        timeout: float | None = None,
    ) -> LLMResult:
        corpus_tokens = sum(count_tokens(s.text) for s in corpus)
        if corpus_tokens > CORPUS_TOKEN_LIMIT:
            log.error("llm call refused: corpus budget", extra={"purpose": purpose, "corpus_tokens": corpus_tokens})
            raise CorpusBudgetError(f"{corpus_tokens} corpus tokens > {CORPUS_TOKEN_LIMIT}")
        prompt = user + render_corpus(corpus)

        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 2):
            started = time.perf_counter()
            try:
                text = self.provider.chat(self.model, system, prompt, max_tokens, timeout or self.timeout)
            except Exception as exc:
                latency = round((time.perf_counter() - started) * 1000, 1)
                log.warning(
                    "llm call failed",
                    extra={
                        "purpose": purpose,
                        "model": self.model,
                        "corpus_tokens": corpus_tokens,
                        "latency_ms": latency,
                        "attempt": attempt,
                        "error": type(exc).__name__,
                    },
                )
                last_exc = exc
                if attempt > self.retries or not _is_retryable(exc):
                    break
                time.sleep(3.0 if "RateLimit" in type(exc).__name__ else 1.0)
                continue
            latency = round((time.perf_counter() - started) * 1000, 1)
            log.info(
                "llm call",
                extra={
                    "purpose": purpose,
                    "model": self.model,
                    "corpus_tokens": corpus_tokens,
                    "latency_ms": latency,
                    "attempt": attempt,
                },
            )
            return LLMResult(text, corpus_tokens, latency)
        raise LLMError(f"{purpose}: {type(last_exc).__name__}") from last_exc

    def complete_json(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        corpus: Sequence[CorpusSegment] = (),
        max_tokens: int = 1024,
        timeout: float | None = None,
        validate=None,
    ) -> dict[str, Any] | None:
        """The reply as a JSON object, or None after one failed repair attempt.
        `validate(obj)` may reject a parsed object (e.g. a Pydantic check)."""

        def ok(obj: dict[str, Any] | None) -> bool:
            if obj is None:
                return False
            if validate is None:
                return True
            try:
                validate(obj)
                return True
            except Exception:
                return False

        first = self.complete(purpose=purpose, system=system, user=user, corpus=corpus, max_tokens=max_tokens, timeout=timeout)
        obj = extract_json_object(first.text)
        if ok(obj):
            return obj
        repair = self.complete(
            purpose=f"{purpose}:repair",
            system=system,
            user=f"{user}\n\n{REPAIR_INSTRUCTION}",
            corpus=corpus,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        obj = extract_json_object(repair.text)
        return obj if ok(obj) else None
