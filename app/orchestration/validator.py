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
- An FAQ or wiki passage with facts is dropped when none of its values is in
  the answer and a higher-authority passage is cited (eval finding: the FAQ's
  outdated "3 days" for bereavement was cited next to the policy's 5, because
  its fact had formed a rule of its own). Alone, it is kept: it may be the only
  source for the point.
- The rule table is code-built from facts, not a document. When the composer
  names it as a source, it stands for the passages of the winning (or upcoming)
  facts whose values are in the answer (`rule_table_sources`); a row named the way
  the table labels it ("rule <key>") stands for that rule's passages. Eval finding:
  an answer taken from the rule table alone ("a PIP lasts 60 calendar days") had
  no citation left and was replaced by "the documents don't cover this".
- Answer text: an outranked value the answer states (and no current or upcoming
  fact states) sends the composer back once to rewrite (`superseded_values`);
  document and chunk ids are always removed from the text (`strip_internal_ids`).
- The route is decided in code from what actually happened, and the response
  is forced to match the §8.1 table (citations / tool_calls empty or not).
- Personal data: phones removed, emails masked, and full names of employees the
  requester may not see are removed from the answer text.
"""

import re
from collections.abc import Iterable, Mapping, Sequence

from app.api.schemas import Citation, ChatResponse, Route, ToolCall
from app.orchestration.composer import RULES_SEGMENT_ID
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


def rule_table_sources(answer: str, resolutions: Sequence[RuleResolution], rule_key: str | None = None) -> list[str]:
    """Passages behind the rule table (or one row of it, `rule_key`): chunks of winning
    or upcoming facts whose value is in the answer. Outranked facts never count."""
    answer_lower = " ".join(answer.lower().split())
    answer_numbers = _numbers(answer)
    out: list[str] = []
    for r in resolutions:
        if rule_key is not None and r.rule_key != rule_key:
            continue
        for f in r.winners + r.upcoming:
            if f.chunk_id not in out and _value_in_answer(f.value, answer_lower, answer_numbers):
                out.append(f.chunk_id)
    return out


def grounded_chunk_ids(
    used_chunk_ids: Iterable[str],
    answer: str,
    resolutions: Sequence[RuleResolution],
    low_authority_values: Mapping[str, Sequence[str]] | None = None,
) -> list[str]:
    """The used passages that actually support the answer (see module doc).
    `low_authority_values`: FAQ / wiki chunk id → the values of its facts."""
    answer_lower = " ".join(answer.lower().split())
    answer_numbers = _numbers(answer)
    supporting: dict[str, list[str]] = {}  # chunk → its winning / upcoming values
    outranked: set[str] = set()  # chunks holding at least one outranked fact
    for r in resolutions:
        for f in r.winners + r.upcoming:
            supporting.setdefault(f.chunk_id, []).append(f.value)
        outranked.update(f.chunk_id for f in r.outranked)
    rule_keys = {r.rule_key for r in resolutions}
    kept = []
    for chunk_id in used_chunk_ids:
        if chunk_id in kept:
            continue
        # A row of the rule table, named as the table lists it ("rule <key>") or by its key.
        row = chunk_id.removeprefix("rule ").strip()
        if chunk_id == RULES_SEGMENT_ID:
            kept += [c for c in rule_table_sources(answer, resolutions) if c not in kept]
        elif row in rule_keys:
            kept += [c for c in rule_table_sources(answer, resolutions, row) if c not in kept]
        elif chunk_id not in outranked:
            kept.append(chunk_id)  # nothing superseded in it: trust the composer's choice
        elif any(_value_in_answer(v, answer_lower, answer_numbers) for v in supporting.get(chunk_id, [])):
            kept.append(chunk_id)
    low = low_authority_values or {}
    if any(c not in low for c in kept):  # a higher-authority passage is cited
        kept = [
            c for c in kept
            if not low.get(c) or any(_value_in_answer(v, answer_lower, answer_numbers) for v in low[c])
        ]
    return kept


# --- answer text checks ------------------------------------------------------
# The composer is told to state current values only and never to name document ids;
# these check that in code, from this request's own precedence result and corpus.

_HYPHENS = dict.fromkeys(map(ord, "‐‑‒–—−"), "-")
_HYPHEN_CLASS = "[-‐‑‒–—−]"


def _norm(text: str) -> str:
    return " ".join(text.translate(_HYPHENS).lower().split())


def _value_phrases(fact) -> list[str]:
    """How a fact's value reads in an answer. A bare number ("10") would also match times,
    balances and other rules' numbers, so it counts only with its unit ("10 days")."""
    value = _norm(fact.value)
    if re.search(r"[a-z]", value):
        return [value]
    unit = _norm(fact.unit or "")
    return [value if unit in value else f"{value} {unit}"] if unit else []


def superseded_values(answer: str, resolutions: Sequence[RuleResolution]) -> list[str]:
    """Outranked values the answer states that no current or upcoming fact states.
    Eval finding: India was told "no fixed deadline" and, next to it, "other versions set
    30 June or 31 March"."""
    text = _norm(answer)
    allowed = {p for r in resolutions for f in r.winners + r.upcoming for p in _value_phrases(f)}
    found: list[str] = []
    for r in resolutions:
        for f in r.outranked:
            for phrase in _value_phrases(f):
                if phrase in found or any(phrase in a for a in allowed):
                    continue
                if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text):
                    found.append(phrase)
    return found


def strip_internal_ids(answer: str, doc_ids: Iterable[str]) -> str:
    """Remove document and chunk ids ("ADD-LEAVE-IN::001", any hyphen character) from the
    answer text, with the brackets around them. Citations carry the sources."""
    ids = sorted({d for d in doc_ids if d}, key=len, reverse=True)
    if not ids:
        return answer
    alternatives = "|".join(_HYPHEN_CLASS.join(re.escape(part) for part in d.split("-")) for d in ids)
    ident = rf"(?:{alternatives})(?:::\d+)?"
    text = re.sub(rf"[ \t]*[(\[]\s*{ident}(?:\s*[,;]\s*{ident})*\s*[)\]]", "", answer, flags=re.IGNORECASE)
    text = re.sub(rf"\b{ident}\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"[ \t]+([.,;:])", r"\1", text)


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
