"""The ingested corpus, loaded read-only from cache.db once ingestion finishes.

Access metadata (access level, regions, effective date) lives only here, read
from the manifest at ingestion; the search index holds ids and vectors only.
"""

import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

import numpy as np

from app.llm.embeddings import EMBEDDING_DIM
from app.retrieval.eligibility import DocMeta


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    locator: str
    text: str
    tokens: int
    suspicious: bool


@dataclass(frozen=True)
class FactRecord:
    fact_id: str
    chunk_id: str
    doc_id: str
    rule_key: str
    topic: str
    attribute: str
    value: str
    unit: str | None
    conditions: str | None


@dataclass
class Corpus:
    docs: dict[str, DocMeta]
    titles: dict[str, str]
    chunks: dict[str, ChunkRecord]  # in document order
    facts_by_chunk: dict[str, list[FactRecord]]
    facts_by_rule: dict[str, list[FactRecord]]
    vector_ids: list[str]
    vectors: np.ndarray  # one unit-length row per id in vector_ids

    @classmethod
    def load(cls, cache_db: str) -> "Corpus":
        conn = sqlite3.connect(f"file:{cache_db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            docs, titles = {}, {}
            for r in conn.execute("SELECT * FROM documents"):
                docs[r["doc_id"]] = DocMeta(
                    r["doc_id"],
                    r["doc_type"],
                    tuple(json.loads(r["regions_json"])),
                    date.fromisoformat(r["effective_date"]),
                    r["access_level"],
                )
                titles[r["doc_id"]] = r["title"]
            chunks = {
                r["chunk_id"]: ChunkRecord(
                    r["chunk_id"], r["doc_id"], r["locator"], r["text"], r["tokens"], bool(r["suspicious"])
                )
                for r in conn.execute("SELECT * FROM chunks ORDER BY doc_id, ordinal")
            }
            facts_by_chunk: dict[str, list[FactRecord]] = defaultdict(list)
            facts_by_rule: dict[str, list[FactRecord]] = defaultdict(list)
            for r in conn.execute("SELECT * FROM facts ORDER BY fact_id"):
                fact = FactRecord(
                    r["fact_id"], r["chunk_id"], r["doc_id"], r["rule_key"], r["topic"],
                    r["attribute"], r["value"], r["unit"], r["conditions"],
                )
                facts_by_chunk[fact.chunk_id].append(fact)
                facts_by_rule[fact.rule_key].append(fact)
            rows = conn.execute("SELECT chunk_id, vector FROM chunk_vectors ORDER BY chunk_id").fetchall()
        finally:
            conn.close()
        vector_ids = [r["chunk_id"] for r in rows]
        vectors = (
            np.stack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
            if rows
            else np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        )
        return cls(docs, titles, chunks, dict(facts_by_chunk), dict(facts_by_rule), vector_ids, vectors)

    def chunk_doc(self, chunk_id: str) -> DocMeta:
        return self.docs[self.chunks[chunk_id].doc_id]
