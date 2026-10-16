"""Post-checks before anything leaves the service (TASK.pdf §4.2, §7.4, §8.1).

- Citations come only from passages that were actually sent to the composer
  and that it says it used: used ∩ sent. A citation can therefore never point
  to a document that wasn't retrieved, or that the requester may not read.
- "Cite every document you used and no others" (§4.2): a passage that holds an
  outranked fact is kept only if one of its winning (or upcoming) values
  appears in the answer — so a superseded or lower-precedence source that the
  model listed anyway is dropped. Passages with no outranked facts are kept as
  listed: the answer may rely on a statement the fact extraction didn't capture
  (eval finding: the India addendum's "carried forward days do not expire").
- The route is decided in code from what actually happened, and the response
  is forced to match the §8.1 table (citations / tool_calls empty or not).
- Personal data: phones removed, emails masked, and full names of employees the
  requester may not see are removed from the answer text.
"""

import re
from collections.abc import Iterable, Sequence

from app.api.schemas import Citation, ChatResponse, Route, ToolCall
from app.privacy.masking import redact_contacts
from app.retrieval.corpus import Corpus
from app.retrieval.precedence import RuleResolution, value_key

NAME_REMOVED = "[name removed]"


def _numbers(text: str) -> set[str]:
    """Numbers in text, number words included ("ten" → 10.0), via precedence.value_key."""
    key = value_key(text)
    return set(key[1:].split(",")) if key.startswith("#") else set()


def _value_in_answer(value: str, answer_lower: str, answer_numbers: set[str]) -> bool:
    numbers = _numbers(value)
    if numbers:
        return numbers <= answer_numbers
    return " ".join(value.lower().split()) in answer_lower


def grounded_chunk_ids(
    used_chunk_ids: Iterable[str], answer: str, resolutions: Sequence[RuleResolution]
) -> list[str]:
    """The used passages that actually support the answer (see module doc)."""
    answer_lower = " ".join(answer.lower().split())
    answer_numbers = _numbers(answer)
    supporting: dict[str, list[str]] = {}  # chunk → its winning / upcoming values
    outranked: set[str] = set()  # chunks holding at least one outranked fact
    for r in resolutions:
        for f in r.winners + r.upcoming:
            supporting.setdefault(f.chunk_id, []).append(f.value)
        outranked.update(f.chunk_id for f in r.outranked)
    kept = []
    for chunk_id in used_chunk_ids:
        if chunk_id not in outranked:
            kept.append(chunk_id)  # nothing superseded in it: trust the composer's choice
        elif any(_value_in_answer(v, answer_lower, answer_numbers) for v in supporting.get(chunk_id, [])):
            kept.append(chunk_id)
    return kept


def build_citations(used_chunk_ids: Iterable[str], sent_chunk_ids: Sequence[str], corpus: Corpus) -> list[Citation]:
    sent = set(sent_chunk_ids)
    citations: list[Citation] = []
    seen: set[tuple[str, str]] = set()
    for chunk_id in used_chunk_ids:
        if chunk_id not in sent or chunk_id not in corpus.chunks:
            continue
        chunk = corpus.chunks[chunk_id]
        key = (chunk.doc_id, chunk.locator)
        if key not in seen:
            seen.add(key)
            citations.append(Citation(doc_id=chunk.doc_id, locator=chunk.locator))
    return citations


def scrub_answer(text: str, hidden_names: Iterable[str]) -> str:
    text = redact_contacts(text)
    for name in sorted({n for n in hidden_names if n.strip()}, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(name)}\b", NAME_REMOVED, text, flags=re.IGNORECASE)
    return text


def route_for(citations: Sequence[Citation], tool_calls: Sequence[ToolCall]) -> Route:
    if citations and tool_calls:
        return "hybrid"
    if citations:
        return "rag"
    if tool_calls:
        return "tool"
    return "no_answer"


_EMPTY_ROUTES = {"clarify", "no_answer", "refuse"}


def finalize(
    request_id: str,
    answer: str,
    route: Route,
    citations: Sequence[Citation] = (),
    tool_calls: Sequence[ToolCall] = (),
    pending_confirmation: bool = False,
    hidden_names: Iterable[str] = (),
) -> ChatResponse:
    """Build the response and enforce the route ↔ fields invariants of §8.1."""
    citations, tool_calls = list(citations), list(tool_calls)
    if route in _EMPTY_ROUTES:
        citations, tool_calls, pending_confirmation = [], [], False
    elif route == "rag":
        tool_calls = []
    elif route == "tool":
        citations = []
    expected = route_for(citations, tool_calls) if route not in _EMPTY_ROUTES else route
    if expected != route:
        raise ValueError(f"route {route} does not match citations/tool_calls ({expected})")
    if pending_confirmation and not any(t.status == "pending_confirmation" for t in tool_calls):
        raise ValueError("pending_confirmation without a pending tool call")
    return ChatResponse(
        request_id=request_id,
        answer=scrub_answer(answer, hidden_names),
        route=route,
        citations=citations,
        tool_calls=tool_calls,
        pending_confirmation=pending_confirmation,
    )
