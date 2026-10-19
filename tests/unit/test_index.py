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


def test_faiss_dense_ranking_matches_exact_dot_product():
    """FAISS IndexFlatIP is exact: same ranks as a plain numpy dot product over the allowed rows."""
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((145, 384)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    query = vectors[7] + 0.1 * rng.standard_normal(384).astype(np.float32)
    query /= np.linalg.norm(query)
    ids = [f"DOC::{i:03d}" for i in range(145)]

    class FixedQuery(HashingEmbedder):
        def embed_query(self, text):
            return query

    index = SearchIndex(ids, ["x"] * 145, vectors, FixedQuery())
    rows = np.array(sorted(rng.choice(145, 90, replace=False)), dtype=np.int64)  # the allowed set

    faiss_scores = index._dense_scores("q", rows)
    numpy_scores = vectors[rows] @ query
    assert np.allclose(faiss_scores, numpy_scores, atol=1e-6)
    assert list(np.argsort(-faiss_scores)[:10]) == list(np.argsort(-numpy_scores)[:10])


def test_dense_search_never_scores_a_restricted_nearest_neighbour():
    """The closest vector overall is restricted: it must not appear, and every allowed chunk still ranks."""
    embedder = HashingEmbedder()
    texts = ["sick leave days", "carry forward annual leave", "public holidays"]
    vectors = embedder.embed(texts)
    index = SearchIndex(["HR::000", "POL::000", "CAL::000"], texts, vectors, embedder)
    hits = index.search("sick leave days", {"POL::000", "CAL::000"}, k=10)
    assert {h.chunk_id for h in hits} == {"POL::000", "CAL::000"}
    assert sorted(h.dense_rank for h in hits) == [1, 2]


def test_misaligned_inputs_rejected():
    import pytest

    with pytest.raises(ValueError):
        SearchIndex(["a"], ["x", "y"], np.zeros((1, 384), dtype=np.float32), HashingEmbedder())


def test_a_second_query_is_fused_in_and_an_identical_one_changes_nothing():
    index = make_index()
    assert index.search("carry forward days", TEXTS.keys())[0].chunk_id == "POL::001"
    assert index.search("carry forward days", TEXTS.keys(), also="carry forward days") == index.search(
        "carry forward days", TEXTS.keys()
    )
    fused = index.search("carry forward days", TEXTS.keys(), also="Deepavali", also_weight=10)
    assert fused[0].chunk_id == "CAL::000"
    assert fused[0].bm25_rank is None  # ranks reported are the main query's
