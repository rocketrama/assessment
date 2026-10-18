"""The one way text LLM calls are made (DESIGN §2, TASK.pdf §4.3).

- Corpus text travels separately from instructions, as `CorpusSegment`s. The
  client counts it with cl100k_base and refuses any call above 3,000 corpus
  tokens (must-pass #4), then wraps it in <documents> delimiters as untrusted data.
- Every call is logged with purpose, model, corpus_tokens and latency.
- Temperature 0; one retry on timeouts, rate limits and server errors.
- An optional `deadline` (time.monotonic) caps each attempt to the time left
  for the whole request (§9.1: 30 s per /chat); no attempt starts with less
  than MIN_CALL_SECONDS left, so a slow provider ends in a safe reply, not a
  timed-out request.
- That cap is a wall-clock limit: the provider call runs in a worker thread and
  the client stops waiting at its timeout (+ HARD_CAP_GRACE_SECONDS). The SDK's
  own timeout is not one: eval round 6 had a 35.1 s /chat although every
  attempt was capped at 12 s. An abandoned call finishes in the background and
  its reply is ignored.
- An optional `hedge_after` sends the same call a second time when the first has
  not answered by then, and takes whichever reply comes first, within the same
  cap. A provider stall is not a slow answer: measured planner calls take 2.1 s
  median and 4.3 s at most, yet eval runs had calls hang for the full 12 s, twice
  in a row. Waiting longer doesn't help with a stalled call; a second one does.
- `complete_json` extracts a JSON object from the reply (tolerating code fences
  and <think> blocks) and makes one repair call if it is invalid.
"""

import json
import logging
import re
import time
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Protocol

from app.config import Settings
from app.tokens import count_tokens

log = logging.getLogger("veltrix.llm")

CORPUS_TOKEN_LIMIT = 3000
MIN_CALL_SECONDS = 2.0
HARD_CAP_GRACE_SECONDS = 0.5
# Worker threads for provider calls; abandoned (timed-out) calls keep a thread until the SDK gives up.
_CALLS = ThreadPoolExecutor(max_workers=32, thread_name_prefix="llm-call")
REPAIR_INSTRUCTION = (
    "Your previous reply was not a valid JSON object of the requested shape. "
    "Reply again with only that JSON object and nothing else."
)


class CorpusBudgetError(Exception):
    """A call would carry more than CORPUS_TOKEN_LIMIT tokens of corpus text."""


class LLMError(Exception):
    """The provider failed after retries."""


class DeadlineExceeded(LLMError):
    """Not enough of the request's time budget left to make (another) call."""


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
    """OpenAI, or an OpenAI-compatible endpoint such as NVIDIA or Gemini (LLM_BASE_URL)."""

    def __init__(self, api_key: str, base_url: str | None = None, reasoning_effort: str = "") -> None:
        from importlib import import_module

        OpenAI = import_module("openai").OpenAI

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


# Default endpoint when LLM_BASE_URL is empty; OpenAI itself uses the SDK default.
_DEFAULT_BASE_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openrouter": "https://openrouter.ai/api/v1",
    "mistral": "https://api.mistral.ai/v1",
}


def openai_compatible_endpoint(settings: Settings) -> tuple[str, str | None]:
    """(API key, base URL) for an OpenAI-compatible provider; the key is "" when unset.
    Shared by text calls and vision OCR."""
    provider = settings.llm_provider
    keys = {
        "openai": settings.openai_api_key,
        "nvidia": settings.nvidia_api_key,
        "gemini": settings.google_api_key,
        "google": settings.google_api_key,
        "openrouter": settings.openrouter_api_key,
        "mistral": settings.mistral_api_key,
    }
    return keys.get(provider, ""), settings.llm_base_url or _DEFAULT_BASE_URLS.get(provider)


class AnthropicProvider:
    def __init__(self, api_key: str) -> None:
        from importlib import import_module

        Anthropic = import_module("anthropic").Anthropic

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


MAX_REPORTED_ERRORS = 5


def describe_validation_error(exc: Exception) -> str:
    """Pydantic errors as "field.path: message"; anything else by its class name.
    Input values are left out on purpose (see complete_json)."""
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            parts = [f"{'.'.join(str(p) for p in e['loc']) or 'reply'}: {e['msg']}" for e in errors()]
        except Exception:
            parts = []
        if parts:
            return "; ".join(parts[:MAX_REPORTED_ERRORS])
    return f"the reply has the wrong shape ({type(exc).__name__})"


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
            key, base_url = openai_compatible_endpoint(settings)
            if not key:
                log.warning("LLM disabled: no API key for provider", extra={"provider": provider_name})
                return None
            provider = OpenAICompatibleProvider(key, base_url, settings.llm_reasoning_effort)
        return cls(provider, settings.llm_model, timeout=settings.llm_timeout_seconds)

    def _call(
        self, purpose: str, system: str, prompt: str, max_tokens: int, call_timeout: float, hedge_after: float | None
    ) -> str:
        """One attempt, capped at call_timeout (+ grace) of wall-clock time; optionally hedged.
        Raises the provider's error when every call sent has failed, TimeoutError at the cap."""
        start = time.monotonic()
        stop = start + call_timeout + HARD_CAP_GRACE_SECONDS
        hedge_at = start + hedge_after if hedge_after and call_timeout - hedge_after >= MIN_CALL_SECONDS else None
        pending: list[Future] = [_CALLS.submit(self.provider.chat, self.model, system, prompt, max_tokens, call_timeout)]
        first_exc: Exception | None = None
        while True:
            until = hedge_at if hedge_at is not None else stop
            done, _ = wait(pending, timeout=max(0.0, until - time.monotonic()), return_when=FIRST_COMPLETED)
            for future in done:
                pending.remove(future)
                try:
                    return future.result()
                except Exception as exc:
                    first_exc = first_exc or exc
            if not pending:  # every call sent failed: the caller's retry logic decides what next
                raise first_exc  # type: ignore[misc]
            now = time.monotonic()
            if hedge_at is not None and now >= hedge_at:
                hedge_at = None
                log.info("llm call hedged", extra={"purpose": purpose, "after_ms": round((now - start) * 1000, 1)})
                left = stop - HARD_CAP_GRACE_SECONDS - now
                pending.append(_CALLS.submit(self.provider.chat, self.model, system, prompt, max_tokens, left))
            elif hedge_at is None and now >= stop:
                raise TimeoutError(f"no reply within {call_timeout:.1f}s")

    def complete(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        corpus: Sequence[CorpusSegment] = (),
        max_tokens: int = 1024,
        timeout: float | None = None,
        deadline: float | None = None,
        hedge_after: float | None = None,
    ) -> LLMResult:
        corpus_tokens = sum(count_tokens(s.text) for s in corpus)
        if corpus_tokens > CORPUS_TOKEN_LIMIT:
            log.error("llm call refused: corpus budget", extra={"purpose": purpose, "corpus_tokens": corpus_tokens})
            raise CorpusBudgetError(f"{corpus_tokens} corpus tokens > {CORPUS_TOKEN_LIMIT}")
        prompt = user + render_corpus(corpus)

        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 2):
            call_timeout = timeout or self.timeout
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining < MIN_CALL_SECONDS:
                    log.warning("llm call skipped: request deadline", extra={"purpose": purpose, "attempt": attempt})
                    raise DeadlineExceeded(purpose) from last_exc
                call_timeout = min(call_timeout, remaining)
            started = time.perf_counter()
            try:
                # Raises TimeoutError (retryable) when the provider overruns its timeout.
                text = self._call(purpose, system, prompt, max_tokens, call_timeout, hedge_after)
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
                pause = 3.0 if "RateLimit" in type(exc).__name__ else 1.0
                if deadline is not None and deadline - time.monotonic() - pause < MIN_CALL_SECONDS:
                    log.warning("llm call skipped: request deadline", extra={"purpose": purpose, "attempt": attempt + 1})
                    raise DeadlineExceeded(purpose) from exc  # no time for a retry: don't wait for it
                time.sleep(pause)
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
        deadline: float | None = None,
        hedge_after: float | None = None,
    ) -> dict[str, Any] | None:
        """The reply as a JSON object, or None after one failed repair attempt.
        `validate(obj)` may reject a parsed object (e.g. a Pydantic check).

        The repair request says what was wrong (field paths and messages, never the
        rejected values: those may quote corpus text). Eval finding: told only "not a
        valid JSON object of the requested shape", the planner repeated the same
        function-call shape ({"name": ..., "args": ...} instead of {"tool": ...})."""
        problem = ""

        def ok(obj: dict[str, Any] | None) -> bool:
            nonlocal problem
            if obj is None:
                problem = "the reply contained no JSON object"
                return False
            if validate is None:
                return True
            try:
                validate(obj)
                return True
            except Exception as exc:
                problem = describe_validation_error(exc)
                return False

        first = self.complete(
            purpose=purpose, system=system, user=user, corpus=corpus, max_tokens=max_tokens, timeout=timeout,
            deadline=deadline, hedge_after=hedge_after,
        )
        obj = extract_json_object(first.text)
        if ok(obj):
            return obj
        log.warning("llm reply rejected", extra={"purpose": purpose, "error": problem})
        repair = self.complete(
            purpose=f"{purpose}:repair",
            system=system,
            user=f"{user}\n\n{REPAIR_INSTRUCTION} What was wrong: {problem}.",
            corpus=corpus,
            max_tokens=max_tokens,
            timeout=timeout,
            deadline=deadline,
            hedge_after=hedge_after,
        )
        obj = extract_json_object(repair.text)
        return obj if ok(obj) else None
