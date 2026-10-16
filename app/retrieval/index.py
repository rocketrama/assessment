"""Hybrid search over the allowed chunks only (DESIGN §2, TASK.pdf §7.2).

BM25 catches exact terms (holiday names, "carry forward", codes); dense
embeddings catch paraphrases ("time off when I'm ill" → sick leave). The two
rankings are merged with Reciprocal Rank Fusion (k=60): rank-based, nothing to
tune, deterministic (ties broken by chunk_id).

Both retrievers see only `allowed_ids`: a restricted chunk is never scored,
never ranked and never returned. BM25 statistics are computed over the allowed
set too, so restricted text can't even influence term weights.

Dense search is an exact dot product in numpy: at this corpus size it returns
the same results as a FAISS IndexFlatIP with nothing extra to install. At
scale (DESIGN §7) this file is the one to swap for a vector database.
"""

import math
import re
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass

import numpy as np

RRF_K = 60
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
        self._vectors = vectors
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

    def search(self, query: str, allowed_ids: Collection[str], k: int = 10) -> list[SearchHit]:
        rows = np.array(sorted(self._row[c] for c in allowed_ids if c in self._row), dtype=np.int64)
        if not len(rows):
            return []
        ids = [self._ids[r] for r in rows]

        bm25 = self._bm25(tokenize(query), rows)
        bm25_order = sorted((i for i in range(len(rows)) if bm25[i] > 0), key=lambda i: (-bm25[i], ids[i]))
        bm25_rank = {i: rank for rank, i in enumerate(bm25_order, start=1)}

        dense = self._vectors[rows] @ self._embedder.embed_query(query)
        dense_order = sorted(range(len(rows)), key=lambda i: (-float(dense[i]), ids[i]))
        dense_rank = {i: rank for rank, i in enumerate(dense_order, start=1)}

        fused = []
        for i in range(len(rows)):
            score = 1 / (RRF_K + dense_rank[i]) + (1 / (RRF_K + bm25_rank[i]) if i in bm25_rank else 0.0)
            fused.append(SearchHit(ids[i], score, bm25_rank.get(i), dense_rank[i]))
        fused.sort(key=lambda h: (-h.score, h.chunk_id))
        return fused[:k]
