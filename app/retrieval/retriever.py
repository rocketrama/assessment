"""Question + requester → passages, rule resolutions and a budgeted context.

1. Eligibility first: only chunks the requester may use on the reference date
   are searched (plus readable, same-region future chunks, which can only ever
   surface as upcoming changes). Restricted chunks are never searched (§7.2).
2. Hybrid search (index.py) over that set.
3. Every rule touched by a hit is resolved across *all* its facts (§5.1) — so
   the winning policy is found even when only an outdated FAQ matched the words.
4. Context in priority order: winning passages (both sides of a conflict),
   upcoming-change passages, then other hits. Passages whose facts all lost
   are left out, so an outdated source can't be cited (§4.2).
"""

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
    ) -> RetrievalResult:
        """`reference_date` is today unless the question is about a past date (§5);
        upcoming changes are only announced for questions about today."""
        corpus, role, region = self.corpus, requester.role, requester.region
        include_upcoming = reference_date >= (today or reference_date)
        allowed = allowed_chunk_ids(self._chunk_docs, corpus.docs, role, region, reference_date)
        upcoming = (
            upcoming_chunk_ids(self._chunk_docs, corpus.docs, role, region, reference_date)
            if include_upcoming
            else frozenset()
        )

        hits = self.index.search(query, allowed | upcoming, k=k)
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
        ordered = self._context_order(current_hits, resolutions, allowed, upcoming)
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
        winning = [f.chunk_id for r in resolutions for f in r.winners]
        coming = [f.chunk_id for r in resolutions for f in r.upcoming]
        lost_facts = {f.fact_id for r in resolutions for f in r.outranked}
        others = []
        for hit in hits:
            facts = self.corpus.facts_by_chunk.get(hit.chunk_id, [])
            # A chunk whose every fact lost precedence must not be cited (§5.1).
            if facts and all(f.fact_id in lost_facts for f in facts):
                continue
            others.append(hit.chunk_id)
        ordered = winning + coming + others
        # Defence in depth (not `assert`, which -O strips): nothing outside the
        # eligible set reaches the context, except upcoming-change passages.
        if any(c not in allowed for c in winning + others) or any(c not in upcoming for c in coming):
            raise RuntimeError("context contains a chunk the requester may not use")
        return ordered
