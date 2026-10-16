"""Hybrid search: BM25 + dense + RRF, strictly inside the allowed set (TASK.pdf §7.2)."""

import numpy as np

from app.retrieval.index import SearchIndex, tokenize
from tests.conftest import HashingEmbedder

TEXTS = {
    "POL::000": "Sick leave: employees receive 12 working days of paid sick leave per year.",
    "POL::001": "Carry forward: up to 10 days of annual leave may be carried forward.",
    "CAL::000": "Public holidays: Deepavali, Republic Day, Christmas Day.",
    "HR::000": "Salary bands per level: L1, L2, L3. Sick leave is not part of pay.",
}


def make_index():
    embedder = HashingEmbedder()
    ids = list(TEXTS)
    return SearchIndex(ids, [TEXTS[i] for i in ids], embedder.embed([TEXTS[i] for i in ids]), embedder)


def test_finds_the_relevant_chunk_first():
    hits = make_index().search("how many sick leave days", TEXTS.keys())
    assert hits[0].chunk_id == "POL::000"


def test_exact_term_found_by_bm25():
    hit = next(h for h in make_index().search("Deepavali", TEXTS.keys()) if h.chunk_id == "CAL::000")
    assert hit.bm25_rank == 1


def test_never_returns_chunks_outside_allowed_ids():
    allowed = {"POL::000", "POL::001", "CAL::000"}
    hits = make_index().search("salary bands per level L1 L2 L3", allowed, k=10)
    assert {h.chunk_id for h in hits} <= allowed
    assert "HR::000" not in {h.chunk_id for h in hits}


def test_empty_or_unknown_allowed_set():
    index = make_index()
    assert index.search("sick", []) == []
    assert index.search("sick", ["NOPE::000"]) == []


def test_bm25_statistics_use_only_allowed_chunks():
    """Same query, same allowed set → same scores whatever else is indexed."""
    index = make_index()
    small = SearchIndex(
        ["POL::000", "POL::001"],
        [TEXTS["POL::000"], TEXTS["POL::001"]],
        HashingEmbedder().embed([TEXTS["POL::000"], TEXTS["POL::001"]]),
        HashingEmbedder(),
    )
    a = index.search("sick leave", {"POL::000", "POL::001"})
    b = small.search("sick leave", {"POL::000", "POL::001"})
    assert [(h.chunk_id, round(h.score, 9)) for h in a] == [(h.chunk_id, round(h.score, 9)) for h in b]


def test_deterministic_ordering_and_k():
    index = make_index()
    first = index.search("leave", TEXTS.keys(), k=2)
    assert len(first) == 2 and first == index.search("leave", TEXTS.keys(), k=2)


def test_tokenize_drops_stopwords_and_keeps_codes():
    assert tokenize("How many days for L5 in the U.S.?") == ["many", "days", "l5", "u.s"]


def test_misaligned_inputs_rejected():
    import pytest

    with pytest.raises(ValueError):
        SearchIndex(["a"], ["x", "y"], np.zeros((1, 384), dtype=np.float32), HashingEmbedder())
