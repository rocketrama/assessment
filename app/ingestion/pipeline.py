"""Ingestion entry point: manifest → load (with OCR) → chunk → injection scan → cache.db.

Runs at startup; `/health` stays 503 until it returns. One bad document is
logged and skipped instead of failing the whole startup (must-pass #1). Holiday
rows are extracted from `calendar` documents. Facts and the search index are
added in later steps.
"""

import hashlib
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass, field, replace

from app.config import Settings
from app.ingestion.cache import cache_db_path, connect_cache, get_parsed, put_parsed, replace_corpus
from app.ingestion.chunker import Chunk, chunk_document
from app.ingestion.extract_holidays import Holiday, extract_holidays
from app.ingestion.injection_scan import looks_like_injection
from app.ingestion.loaders import load_document
from app.ingestion.loaders.ocr import PageOcr, tesseract_available
from app.ingestion.manifest import ManifestEntry, ManifestError, load_manifest
from app.llm.vision import VisionTranscriber

log = logging.getLogger("veltrix.ingestion")

# Bump when loader or chunker output changes, so cached parses are not reused.
PARSER_VERSION = "1"


class IngestionError(Exception):
    pass


@dataclass
class IngestionReport:
    documents: int = 0
    chunks: int = 0
    corpus_tokens: int = 0
    max_chunk_tokens: int = 0
    cache_hits: int = 0
    skipped: list[str] = field(default_factory=list)  # "doc_id: reason"
    failed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ocr_pages: Counter = field(default_factory=Counter)  # engine → chunks
    suspicious_chunks: list[str] = field(default_factory=list)
    low_confidence_chunks: list[str] = field(default_factory=list)
    holidays: Counter = field(default_factory=Counter)  # doc_id → office-closed days
    seconds: float = 0.0


def build_page_ocr(settings: Settings) -> PageOcr:
    use_tesseract = tesseract_available(settings.tesseract_cmd)
    if not use_tesseract:
        log.warning("tesseract not available; scanned pages depend on the vision model")
    return PageOcr(
        vision=VisionTranscriber.from_settings(settings),
        use_tesseract=use_tesseract,
        min_confidence=settings.ocr_min_confidence,
    )


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def run_ingestion(settings: Settings, ocr: PageOcr | None = None) -> IngestionReport:
    started = time.perf_counter()
    try:
        entries, skipped = load_manifest(settings.manifest_path, settings.corpus_dir)
    except ManifestError as exc:
        raise IngestionError(str(exc)) from exc

    report = IngestionReport(skipped=[f"{s.doc_id or f'#{s.index}'}: {s.reason}" for s in skipped])
    ocr = ocr or build_page_ocr(settings)
    documents: list[tuple[ManifestEntry, str]] = []
    all_chunks: list[Chunk] = []
    all_holidays: list[Holiday] = []

    with connect_cache(cache_db_path(settings.cache_dir)) as cache:
        for entry in entries:
            path = os.path.join(settings.corpus_dir, entry.file)
            sha = _sha256(path)
            cache_key = f"{sha}:{PARSER_VERSION}:{ocr.engines()}"
            parsed = get_parsed(cache, cache_key)
            if parsed is not None:
                report.cache_hits += 1
            else:
                try:
                    parsed = load_document(path, ocr)
                except Exception as exc:
                    report.failed.append(f"{entry.doc_id}: {type(exc).__name__}")
                    log.exception("document failed to load", extra={"doc_id": entry.doc_id})
                    continue
                if not parsed.degraded:
                    put_parsed(cache, cache_key, entry.doc_id, parsed)

            report.warnings.extend(f"{entry.doc_id}: {w}" for w in parsed.warnings)
            chunks = [_flag(c) for c in chunk_document(entry.doc_id, parsed.blocks)]
            if not chunks:
                report.warnings.append(f"{entry.doc_id}: no text extracted")
            documents.append((entry, sha))
            all_chunks.extend(chunks)
            if entry.doc_type == "calendar":
                holidays, warnings = extract_holidays(entry.doc_id, entry.regions, parsed.blocks)
                report.warnings.extend(f"{entry.doc_id}: {w}" for w in warnings)
                if not any(h.office_closed for h in holidays):
                    report.warnings.append(f"{entry.doc_id}: calendar has no holiday table")
                all_holidays.extend(holidays)

        replace_corpus(cache, documents, all_chunks, all_holidays)

    report.documents = len(documents)
    report.chunks = len(all_chunks)
    report.corpus_tokens = sum(c.tokens for c in all_chunks)
    report.max_chunk_tokens = max((c.tokens for c in all_chunks), default=0)
    report.ocr_pages = Counter(c.ocr_engine for c in all_chunks if c.ocr_engine)
    report.suspicious_chunks = [c.chunk_id for c in all_chunks if c.suspicious]
    report.low_confidence_chunks = [c.chunk_id for c in all_chunks if c.low_confidence]
    report.holidays = Counter(h.doc_id for h in all_holidays if h.office_closed)
    report.seconds = round(time.perf_counter() - started, 2)
    _log_report(report)
    return report


def _flag(chunk: Chunk) -> Chunk:
    return replace(chunk, suspicious=True) if looks_like_injection(chunk.text) else chunk


def _log_report(report: IngestionReport) -> None:
    for line in report.skipped + report.failed + report.warnings:
        log.warning("ingestion issue", extra={"detail": line})
    log.info(
        "ingestion report",
        extra={
            "documents": report.documents,
            "chunks": report.chunks,
            "corpus_tokens": report.corpus_tokens,
            "max_chunk_tokens": report.max_chunk_tokens,
            "cache_hits": report.cache_hits,
            "skipped": len(report.skipped),
            "failed": len(report.failed),
            "warnings": len(report.warnings),
            "ocr_chunks": dict(report.ocr_pages),
            "suspicious_chunks": report.suspicious_chunks,
            "low_confidence_chunks": report.low_confidence_chunks,
            "holidays": dict(report.holidays),
            "seconds": report.seconds,
        },
    )
