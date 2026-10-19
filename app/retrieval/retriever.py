"""Question + requester → passages, rule resolutions and a budgeted context.

1. Eligibility first: only chunks the requester may use on the reference date
   are searched (plus readable, same-region future chunks, which can only ever
   surface as upcoming changes). Restricted chunks are never searched (§7.2).
2. Hybrid search (index.py) over that set.
3. Every rule touched by a hit is resolved across *all* its facts (§5.1) — so
   the winning policy is found even when only an outdated FAQ matched the words.
4. Context in hit-rank order: at each hit, first the winning passages (both
   sides of a conflict) and upcoming-change passages of the rules that hit
   raised, then the hit itself. Passages whose facts all lost are left out, so
   an outdated source can't be cited (§4.2). RAGAS finding: with every rule's
   winners put first, the 8th hit's rules (remote work, carry forward) filled
   the context ahead of the 1st hit, the holiday calendar that answered the
   question (context precision 0.06-0.11), and pushed the manager handbook out
   of the budget for "how quickly must I respond to a leave request".
5. A passage that names another document by its manifest title ("according to
   the Annual Leave Accrual Schedule") brings that document's passages in right
   after it, within the requester's eligible set and a token cap. Eval finding:
   the accrual table is mostly numbers and ranked 3rd even within its own
   document, so "how many days of annual leave do I get?" had the policy's
   pointer to the schedule but not the 18/21/24 days themselves.
"""

import re

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from app.data.employees import Employee
from app.retrieval.context_packer import PackedContext, pack_context
from app.retrieval.corpus import Corpus
from app.retrieval.eligibility import allowed_chunk_ids, upcoming_chunk_ids
from app.retrieval.index import SearchHit, SearchIndex
from app.retrieval.precedence import RuleResolution, resolve

DEFAULT_TOP_K = 10
# Rules are collected in hit-rank order; the first few are the ones the question
# is about. More would only add off-topic rules to the answer's decision table.
MAX_RULES = 8
# A referenced document adds at most this many corpus tokens (the accrual
# schedule is ~360), and at most this many documents are followed per question.
LINKED_DOC_MAX_TOKENS = 600
MAX_LINKED_DOCS = 2


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def title_links(chunks, titles) -> dict[str, tuple[str, ...]]:
    """chunk_id → the other documents whose exact title the chunk mentions."""
    normalized = {doc_id: f" {_norm(t)} " for doc_id, t in titles.items() if _norm(t)}
    links: dict[str, tuple[str, ...]] = {}
    for chunk in chunks:
        text = f" {_norm(chunk.text)} "
        found = tuple(sorted(d for d, t in normalized.items() if d != chunk.doc_id and t in text))
        if found:
            links[chunk.chunk_id] = found
    return links


@dataclass(frozen=True)
class RetrievalResult:
    reference_date: date
    hits: tuple[SearchHit, ...]  # current-eligible hits only
    resolutions: tuple[RuleResolution, ...]
    ordered_chunk_ids: tuple[str, ...]  # priority order, before packing
    context: PackedContext  # packed with the full budget (the composer repacks around the rule table)
    allowed_chunks: int


class Retriever:
    def __init__(self, corpus: Corpus, embedder) -> None:
        self.corpus = corpus
        texts = {c.chunk_id: c.text for c in corpus.chunks.values()}
        self.index = SearchIndex(corpus.vector_ids, [texts[i] for i in corpus.vector_ids], corpus.vectors, embedder)
        self._chunk_docs = {c.chunk_id: c.doc_id for c in corpus.chunks.values()}
        self._links = title_links(corpus.chunks.values(), corpus.titles)

    @classmethod
    def load(cls, cache_db: str, embedder) -> "Retriever":
        return cls(Corpus.load(cache_db), embedder)

    def retrieve(
        self,
        query: str,
        requester: Employee,
        reference_date: date,
        today: date | None = None,
        k: int = DEFAULT_TOP_K,
        message: str | None = None,
    ) -> RetrievalResult:
        """`reference_date` is today unless the question is about a past date (§5);
        upcoming changes are only announced for questions about today.
        `message`: the employee's own words when `query` is the planner's rewrite;
        searched too, at half weight. RAGAS finding: "how quickly must I respond to a
        leave request from my team?" ranks the manager handbook 2nd, the planner's
        "response time for manager on leave requests" only 5th to beyond the top 10."""
        corpus, role, region = self.corpus, requester.role, requester.region
        include_upcoming = reference_date >= (today or reference_date)
        allowed = allowed_chunk_ids(self._chunk_docs, corpus.docs, role, region, reference_date)
        upcoming = (
            upcoming_chunk_ids(self._chunk_docs, corpus.docs, role, region, reference_date)
            if include_upcoming
            else frozenset()
        )

        hits = self.index.search(query, allowed | upcoming, k=k, also=message)
        current_hits = tuple(h for h in hits if h.chunk_id in allowed)

        rule_keys: list[str] = []
        for hit in hits:
            for fact in corpus.facts_by_chunk.get(hit.chunk_id, []):
                if fact.rule_key not in rule_keys and len(rule_keys) < MAX_RULES:
                    rule_keys.append(fact.rule_key)
        resolutions = tuple(
            r
            for key in rule_keys
            if (r := resolve(key, corpus.facts_by_rule[key], corpus.docs, role, region, reference_date, include_upcoming))
        )
        ordered = self._context_order(hits, resolutions, allowed, upcoming)
        ordered = self._with_linked_docs(
            query, ordered, {h.chunk_id for h in current_hits}, allowed, role, region, reference_date
        )
        return RetrievalResult(
            reference_date, current_hits, resolutions, tuple(ordered), pack_context(ordered, corpus.chunks), len(allowed)
        )

    def _context_order(
        self,
        hits: Sequence[SearchHit],
        resolutions: Sequence[RuleResolution],
        allowed: frozenset[str],
        upcoming: frozenset[str],
    ) -> list[str]:
        """`hits` include upcoming-only hits: they place their rules' passages but are
        not added themselves (an upcoming passage enters only as a rule's upcoming fact)."""
        pending = {r.rule_key: r for r in resolutions}
        lost_facts = {f.fact_id for r in resolutions for f in r.outranked}
        ordered: list[str] = []
        current: list[str] = []
        coming: list[str] = []
        for hit in hits:
            facts = self.corpus.facts_by_chunk.get(hit.chunk_id, [])
            for fact in facts:
                if (r := pending.pop(fact.rule_key, None)) is not None:
                    winning = [f.chunk_id for f in r.winners]
                    ordered += winning + [f.chunk_id for f in r.upcoming]
                    current += winning
                    coming += [f.chunk_id for f in r.upcoming]
            # A chunk whose every fact lost precedence must not be cited (§5.1).
            if hit.chunk_id not in allowed or (facts and all(f.fact_id in lost_facts for f in facts)):
                continue
            ordered.append(hit.chunk_id)
            current.append(hit.chunk_id)
        # Defence in depth (not `assert`, which -O strips): nothing outside the
        # eligible set reaches the context, except upcoming-change passages.
        if any(c not in allowed for c in current) or any(c not in upcoming for c in coming):
            raise RuntimeError("context contains a chunk the requester may not use")
        return ordered

    def _with_linked_docs(
        self,
        query: str,
        ordered: Sequence[str],
        hit_ids: set[str],
        allowed: frozenset[str],
        role: str,
        region: str,
        reference_date: date,
    ) -> list[str]:
        """Insert the passages of documents named by a search hit right after it:
        eligible chunks only, best search match first, up to LINKED_DOC_MAX_TOKENS each.
        A linked passage whose every fact lost precedence is skipped, as in step 4.
        Only hits link: a passage that is here because one of its facts won a rule
        (the v2 summary naming "Global Leave Policy (Version 2)") pulled in v2's
        sick, bereavement and review sections for an annual-leave question."""
        corpus = self.corpus
        present = set(ordered)
        followed: list[str] = []
        out: list[str] = []
        for chunk_id in ordered:
            out.append(chunk_id)
            for doc_id in self._links.get(chunk_id, ()) if chunk_id in hit_ids else ():
                if doc_id in followed or len(followed) >= MAX_LINKED_DOCS:
                    continue
                ids = [c for c, d in self._chunk_docs.items() if d == doc_id and c in allowed]
                if not ids:  # restricted, other region or not yet in effect
                    continue
                followed.append(doc_id)
                used = 0
                for hit in self.index.search(query, ids, k=len(ids)):
                    chunk = corpus.chunks[hit.chunk_id]
                    if hit.chunk_id in present or used + chunk.tokens > LINKED_DOC_MAX_TOKENS:
                        continue
                    if self._all_facts_lost(hit.chunk_id, role, region, reference_date):
                        continue
                    out.append(hit.chunk_id)
                    present.add(hit.chunk_id)
                    used += chunk.tokens
        if any(c not in allowed for c in out if c not in ordered):
            raise RuntimeError("context contains a chunk the requester may not use")
        return out

    def _all_facts_lost(self, chunk_id: str, role: str, region: str, reference_date: date) -> bool:
        facts = self.corpus.facts_by_chunk.get(chunk_id, [])
        if not facts:
            return False
        for fact in facts:
            r = resolve(fact.rule_key, self.corpus.facts_by_rule[fact.rule_key], self.corpus.docs,
                        role, region, reference_date, include_upcoming=False)
            if r is not None and fact in r.winners:
                return False
        return True
