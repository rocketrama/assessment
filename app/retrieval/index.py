"""Hybrid search over the allowed chunks only (DESIGN §2, TASK.pdf §7.2).

BM25 catches exact terms (holiday names, "carry forward", codes); dense
embeddings catch paraphrases ("time off when I'm ill" → sick leave). The two
rankings are merged with weighted Reciprocal Rank Fusion (k=60): rank-based,
deterministic (ties broken by chunk_id). BM25 counts half: when the question
shares no words with the right passage ("my grandmother passed away" vs
"bereavement"), BM25 still ranks weak matches on common words ("home"), and at
equal weight those pushed the vector search's top two hits down to #11 and #12
(measured on the assessment corpus; at half weight they stay in the top 10).

Both retrievers see only `allowed_ids`: a restricted chunk is never scored,
never ranked and never returned. BM25 statistics are computed over the allowed
set too, so restricted text can't even influence term weights.

Dense search is a FAISS IndexFlatIP (exact inner product = cosine on unit
vectors), searched with an IDSelectorBatch of the allowed rows: the access
filter is applied inside the search, never after it. The index is built in
memory from the vectors in SQLite at startup (one source of truth; no index
file to go stale). At scale (DESIGN §7) the flat index becomes IVF/HNSW.
"""

import math
import re
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass

import faiss
import numpy as np

RRF_K = 60
DENSE_WEIGHT = 1.0
BM25_WEIGHT = 0.5
BM25_K1 = 1.2
BM25_B = 0.75

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i in is it its my of on or "
    "the their there this to was what when where which who will with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    score: float  # RRF score
    bm25_rank: int | None  # 1-based; None when no query term occurs in the chunk
    dense_rank: int


class SearchIndex:
    def __init__(self, ids: list[str], texts: list[str], vectors: np.ndarray, embedder) -> None:
        if len(ids) != len(texts) or len(ids) != len(vectors):
            raise ValueError("ids, texts and vectors must align")
        self._ids = ids
        self._row = {chunk_id: i for i, chunk_id in enumerate(ids)}
        self._terms = [Counter(tokenize(t)) for t in texts]
        self._lengths = np.array([sum(c.values()) for c in self._terms], dtype=np.float64)
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._dense = faiss.IndexFlatIP(vectors.shape[1] if vectors.ndim == 2 else 1)
        if len(ids):
            self._dense.add(vectors)  # FAISS row i = self._ids[i]
        self._embedder = embedder

    def __len__(self) -> int:
        return len(self._ids)

    def _bm25(self, query_terms: list[str], rows: np.ndarray) -> np.ndarray:
        """Okapi BM25 with the always-positive Lucene IDF, over `rows` only."""
        scores = np.zeros(len(rows))
        if not query_terms or not len(rows):
            return scores
        n_docs = len(rows)
        avg_len = float(self._lengths[rows].mean()) or 1.0
        norm = BM25_K1 * (1 - BM25_B + BM25_B * self._lengths[rows] / avg_len)
        for term in set(query_terms):
            tf = np.array([self._terms[r].get(term, 0) for r in rows], dtype=np.float64)
            df = int((tf > 0).sum())
            if df == 0:
                continue
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            scores += idf * tf * (BM25_K1 + 1) / (tf + norm)
        return scores

    def _dense_scores(self, query: str, rows: np.ndarray) -> np.ndarray:
        """Cosine score of every allowed row, aligned with `rows`. FAISS searches only the
        selected rows; k = len(rows), so every allowed row gets a rank."""
        q = np.ascontiguousarray(self._embedder.embed_query(query), dtype=np.float32).reshape(1, -1)
        params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(rows))
        scores, found = self._dense.search(q, len(rows), params=params)
        position = {int(r): i for i, r in enumerate(rows)}
        dense = np.full(len(rows), -np.inf)
        for score, row in zip(scores[0], found[0]):
            if row >= 0:  # -1 = no more results
                dense[position[int(row)]] = score
        return dense

    def search(self, query: str, allowed_ids: Collection[str], k: int = 10) -> list[SearchHit]:
        rows = np.array(sorted(self._row[c] for c in allowed_ids if c in self._row), dtype=np.int64)
        if not len(rows):
            return []
        ids = [self._ids[r] for r in rows]

        bm25 = self._bm25(tokenize(query), rows)
        bm25_order = sorted((i for i in range(len(rows)) if bm25[i] > 0), key=lambda i: (-bm25[i], ids[i]))
        bm25_rank = {i: rank for rank, i in enumerate(bm25_order, start=1)}

        dense = self._dense_scores(query, rows)
        dense_order = sorted(range(len(rows)), key=lambda i: (-float(dense[i]), ids[i]))
        dense_rank = {i: rank for rank, i in enumerate(dense_order, start=1)}

        fused = []
        for i in range(len(rows)):
            score = DENSE_WEIGHT / (RRF_K + dense_rank[i])
            if i in bm25_rank:
                score += BM25_WEIGHT / (RRF_K + bm25_rank[i])
            fused.append(SearchHit(ids[i], score, bm25_rank.get(i), dense_rank[i]))
        fused.sort(key=lambda h: (-h.score, h.chunk_id))
        return fused[:k]
