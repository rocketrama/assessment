"""Context order: a rule's winning passage goes where the hit that raised the rule ranks,
not ahead of better hits (RAGAS context precision on holiday questions)."""

from datetime import date

import numpy as np

from app.data.employees import Employee
from app.retrieval.corpus import ChunkRecord, Corpus, FactRecord
from app.retrieval.eligibility import DocMeta
from app.retrieval.retriever import Retriever
from tests.conftest import HashingEmbedder

TODAY = date(2026, 9, 15)
EMPLOYEE = Employee("E1", "IN", "employee", "L2", None, "2024-01-01")

TEXTS = {
    "CAL::000": ("CAL", "Deepavali holidays calendar: Deepavali 9 November."),
    "FAQ::000": ("FAQ", "Deepavali FAQ: sick days are 10 per year."),
    "POL::000": ("POL", "Sick leave policy: twelve working days per year."),
}
TYPES = {"CAL": "calendar", "FAQ": "faq", "POL": "policy"}
FACTS = [
    FactRecord("f1", "FAQ::000", "FAQ", "sick-days", "sick leave", "days per year", "10", None, None),
    FactRecord("f2", "POL::000", "POL", "sick-days", "sick leave", "days per year", "12", None, None),
]


def make_retriever():
    chunks = {cid: ChunkRecord(cid, doc, cid, text, 20, False) for cid, (doc, text) in TEXTS.items()}
    docs = {d: DocMeta(d, t, ("GLOBAL",), date(2025, 1, 1), "all") for d, t in TYPES.items()}
    ids = sorted(chunks)
    vectors = HashingEmbedder().embed([chunks[i].text for i in ids]).astype(np.float32)
    by_chunk = {"FAQ::000": [FACTS[0]], "POL::000": [FACTS[1]]}
    corpus = Corpus(docs, {d: d for d in docs}, chunks, by_chunk, {"sick-days": FACTS}, ids, vectors)
    return Retriever(corpus, HashingEmbedder())


def test_winning_passage_follows_the_hit_that_raised_its_rule():
    result = make_retriever().retrieve("Deepavali holidays", EMPLOYEE, TODAY, today=TODAY, k=2)
    assert [h.chunk_id for h in result.hits] == ["CAL::000", "FAQ::000"]
    # The calendar hit stays first; the policy that outranks the FAQ takes the FAQ's place.
    assert list(result.ordered_chunk_ids) == ["CAL::000", "POL::000"]
