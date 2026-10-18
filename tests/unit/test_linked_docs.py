"""A passage that names another document by title brings that document's passages in."""

from datetime import date

import numpy as np

from app.data.employees import Employee
from app.retrieval.corpus import ChunkRecord, Corpus
from app.retrieval.eligibility import DocMeta
from app.retrieval.retriever import LINKED_DOC_MAX_TOKENS, Retriever, title_links
from tests.conftest import HashingEmbedder

TODAY = date(2026, 9, 15)
IN_EMPLOYEE = Employee("E1", "IN", "employee", "L2", None, "2024-01-01")

TEXTS = {
    "POL::000": ("POL", "Annual leave. Employees earn annual leave each month according to the Annual Leave Accrual Schedule."),
    "SCHED::000": ("SCHED", "Accrual rates. Less than 2 years: 18. 2 to 5 years: 21. 5 years or more: 24."),
    "SCHED::001": ("SCHED", "Worked example of how accrual works for one employee."),
    "HR::000": ("HR", "Pay bands for staff are set out in the Annual Leave Accrual Schedule appendix."),
    "SECRET::000": ("SECRET", "Confidential numbers."),
    "POL2::000": ("POL2", "Holiday pay follows the Secret Bands document."),
}
TITLES = {"POL": "Leave Policy", "SCHED": "Annual Leave Accrual Schedule", "HR": "HR Notes",
          "SECRET": "Secret Bands", "POL2": "Holiday Policy"}
ACCESS = {"SECRET": "hr"}


def make_retriever(tokens=None):
    tokens = tokens or {}
    chunks = {cid: ChunkRecord(cid, doc, cid, text, tokens.get(cid, 30), False) for cid, (doc, text) in TEXTS.items()}
    docs = {d: DocMeta(d, "policy", ("GLOBAL",), date(2025, 1, 1), ACCESS.get(d, "all")) for d in TITLES}
    ids = sorted(chunks)
    vectors = HashingEmbedder().embed([chunks[i].text for i in ids]).astype(np.float32)
    return Retriever(Corpus(docs, TITLES, chunks, {}, {}, ids, vectors), HashingEmbedder())


def test_title_links_find_exact_titles_of_other_documents_only():
    retriever = make_retriever()
    links = title_links(retriever.corpus.chunks.values(), TITLES)
    assert links["POL::000"] == ("SCHED",)
    assert "SCHED::000" not in links  # its own title is not a link


def test_named_document_follows_the_passage_that_names_it():
    """k=1: only the policy passage is a search hit; the schedule comes in through its title."""
    result = make_retriever().retrieve("How many days of annual leave do I earn?", IN_EMPLOYEE, TODAY, today=TODAY, k=1)
    ordered = list(result.ordered_chunk_ids)
    assert [h.chunk_id for h in result.hits] == ["POL::000"]
    assert "SCHED::000" in ordered and "SCHED::000" in result.context.chunk_ids
    assert ordered.index("POL::000") < ordered.index("SCHED::000")


def test_a_named_document_the_requester_may_not_read_is_never_linked():
    result = make_retriever().retrieve("holiday pay follows", IN_EMPLOYEE, TODAY, today=TODAY, k=1)
    assert [h.chunk_id for h in result.hits] == ["POL2::000"]
    assert not any(c.startswith("SECRET::") for c in result.ordered_chunk_ids)


def test_linked_passages_stay_within_the_token_cap():
    big = {"SCHED::000": LINKED_DOC_MAX_TOKENS, "SCHED::001": LINKED_DOC_MAX_TOKENS}
    result = make_retriever(big).retrieve("How many days of annual leave do I earn?", IN_EMPLOYEE, TODAY, today=TODAY, k=1)
    assert [h.chunk_id for h in result.hits] == ["POL::000"]
    assert len([c for c in result.ordered_chunk_ids if c.startswith("SCHED::")]) == 1
