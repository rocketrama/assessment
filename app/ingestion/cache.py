"""The ingestion cache database (`CACHE_DIR/cache.db`), outside ./data.

- `parse_cache`: loader/OCR output per file, keyed by the file's SHA-256 plus the
  parser version and OCR engines — never by file name, so a modified corpus
  file with the same name is always re-read (must-pass #5). Each entry carries a
  checksum; an entry that fails it is recomputed, never loaded.
- `documents` / `chunks` / `holidays` / `facts`: the current corpus, rebuilt in one transaction on every
  startup with manifest metadata read fresh, so a stale cache can never mask an
  access-level change.

Disposable: deleting the file only costs a slower next startup.
"""

import hashlib
import json
import logging
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone

from app.ingestion.blocks import Block, ParsedDocument
from app.ingestion.chunker import Chunk
from app.ingestion.extract_facts import Fact
from app.ingestion.extract_holidays import Holiday
from app.ingestion.manifest import ManifestEntry

log = logging.getLogger("veltrix.ingestion")

SCHEMA_VERSION = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS parse_cache (
    cache_key    TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    checksum     TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    doc_id         TEXT PRIMARY KEY,
    file           TEXT NOT NULL,
    title          TEXT NOT NULL,
    doc_type       TEXT NOT NULL,
    regions_json   TEXT NOT NULL,
    effective_date TEXT NOT NULL,
    access_level   TEXT NOT NULL,
    sha256         TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id          TEXT PRIMARY KEY,
    doc_id            TEXT NOT NULL REFERENCES documents(doc_id),
    ordinal           INTEGER NOT NULL,
    heading_path_json TEXT NOT NULL,
    locator           TEXT NOT NULL,
    page              INTEGER,
    text              TEXT NOT NULL,
    tokens            INTEGER NOT NULL,
    has_table         INTEGER NOT NULL,
    ocr_engine        TEXT,
    low_confidence    INTEGER NOT NULL,
    suspicious        INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id, ordinal);
CREATE TABLE IF NOT EXISTS holidays (
    doc_id        TEXT NOT NULL REFERENCES documents(doc_id),
    region        TEXT NOT NULL,
    date          TEXT NOT NULL,
    name          TEXT NOT NULL,
    office_closed INTEGER NOT NULL,
    PRIMARY KEY (doc_id, region, date, name)
);
CREATE TABLE IF NOT EXISTS facts (
    fact_id    TEXT PRIMARY KEY,
    chunk_id   TEXT NOT NULL REFERENCES chunks(chunk_id),
    doc_id     TEXT NOT NULL REFERENCES documents(doc_id),
    rule_key   TEXT NOT NULL,
    topic      TEXT NOT NULL,
    attribute  TEXT NOT NULL,
    value      TEXT NOT NULL,
    unit       TEXT,
    conditions TEXT,
    quote      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_facts_rule ON facts(rule_key);
CREATE INDEX IF NOT EXISTS idx_facts_chunk ON facts(chunk_id);
"""

_TABLES = ("meta", "parse_cache", "fact_cache", "documents", "chunks", "holidays", "facts")


def cache_db_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "cache.db")


@contextmanager
def connect_cache(path: str) -> Iterator[sqlite3.Connection]:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None, timeout=5.0)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _ensure_schema(conn)
        yield conn
    finally:
        conn.close()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    has_meta = conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'meta'").fetchone()
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone() if has_meta else None
    if row is not None and row["value"] == str(SCHEMA_VERSION):
        return
    if row is not None:
        log.warning("cache schema changed; rebuilding cache", extra={"old": row["value"], "new": SCHEMA_VERSION})
    for table in reversed(_TABLES):
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.executescript(_SCHEMA)
    # Written last: an interrupted rebuild has no version row and is simply redone.
    conn.execute("INSERT INTO meta (key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))


def _checksum(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_parsed(conn: sqlite3.Connection, cache_key: str) -> ParsedDocument | None:
    row = conn.execute(
        "SELECT payload_json, checksum FROM parse_cache WHERE cache_key = ?", (cache_key,)
    ).fetchone()
    if row is None:
        return None
    if _checksum(row["payload_json"]) != row["checksum"]:
        log.warning("cache entry failed checksum; recomputing", extra={"cache_key": cache_key[:16]})
        return None
    try:
        payload = json.loads(row["payload_json"])
        return ParsedDocument([Block.from_dict(b) for b in payload["blocks"]], list(payload["warnings"]))
    except (ValueError, KeyError, TypeError):
        log.warning("cache entry unreadable; recomputing", extra={"cache_key": cache_key[:16]})
        return None


def put_parsed(conn: sqlite3.Connection, cache_key: str, doc_id: str, parsed: ParsedDocument) -> None:
    payload = json.dumps({"blocks": [b.to_dict() for b in parsed.blocks], "warnings": parsed.warnings})
    conn.execute(
        "INSERT OR REPLACE INTO parse_cache (cache_key, doc_id, payload_json, checksum, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (cache_key, doc_id, payload, _checksum(payload), datetime.now(timezone.utc).isoformat()),
    )


def replace_corpus(
    conn: sqlite3.Connection,
    documents: list[tuple[ManifestEntry, str]],
    chunks: list[Chunk],
    holidays: list[Holiday],
    facts: list[Fact] = (),
) -> None:
    """Swap in the current documents, chunks, holidays and facts atomically."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("DELETE FROM facts")
        conn.execute("DELETE FROM holidays")
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM documents")
        conn.executemany(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    e.doc_id,
                    e.file,
                    e.title,
                    e.doc_type,
                    json.dumps(list(e.regions)),
                    e.effective_date.isoformat(),
                    e.access_level,
                    sha,
                )
                for e, sha in documents
            ],
        )
        conn.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    c.chunk_id,
                    c.doc_id,
                    c.ordinal,
                    json.dumps(list(c.heading_path)),
                    c.locator,
                    c.page,
                    c.text,
                    c.tokens,
                    int(c.has_table),
                    c.ocr_engine,
                    int(c.low_confidence),
                    int(c.suspicious),
                )
                for c in chunks
            ],
        )
        conn.executemany(
            "INSERT OR IGNORE INTO holidays VALUES (?, ?, ?, ?, ?)",
            [(h.doc_id, h.region, h.date.isoformat(), h.name, int(h.office_closed)) for h in holidays],
        )
        conn.executemany(
            "INSERT INTO facts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (f.fact_id, f.chunk_id, f.doc_id, f.rule_key, f.topic, f.attribute, f.value, f.unit, f.conditions, f.quote)
                for f in facts
            ],
        )
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
