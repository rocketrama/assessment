"""manifest.json validation (TASK.pdf §2.2).

Bad entries are logged and skipped rather than failing the whole startup, so one
broken line can't keep /health at 503. Access level, regions, dates and type are
read fresh from here on every startup and never cached.
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

log = logging.getLogger("veltrix.ingestion")

DocType = Literal["policy", "addendum", "memo", "handbook", "calendar", "faq", "wiki"]
AccessLevel = Literal["all", "managers", "hr"]
SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".html", ".htm", ".md")


class ManifestEntry(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    doc_id: str = Field(min_length=1, max_length=200)
    file: str = Field(min_length=1, max_length=300)
    title: str
    doc_type: DocType
    regions: tuple[str, ...] = Field(min_length=1)
    effective_date: date
    access_level: AccessLevel

    @field_validator("file")
    @classmethod
    def _plain_file_name(cls, value: str) -> str:
        # A file name inside corpus/, never a path that could escape it.
        if os.path.basename(value) != value or value in (".", "..") or "\\" in value:
            raise ValueError("file must be a plain file name inside corpus/")
        return value

    @field_validator("regions")
    @classmethod
    def _upper_regions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(r.strip().upper() for r in value)


@dataclass(frozen=True)
class SkippedEntry:
    index: int
    doc_id: str | None
    reason: str


class ManifestError(Exception):
    """The manifest as a whole is unusable (missing, not JSON, not an array)."""


def load_manifest(manifest_path: str, corpus_dir: str) -> tuple[list[ManifestEntry], list[SkippedEntry]]:
    if not os.path.isfile(manifest_path):
        raise ManifestError(f"manifest not found: {manifest_path}")
    if not os.path.isdir(corpus_dir):
        raise ManifestError(f"corpus dir not found: {corpus_dir}")
    try:
        with open(manifest_path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"manifest is not valid JSON: {exc}") from exc
    if not isinstance(raw, list):
        raise ManifestError("manifest.json must be a JSON array")

    entries: list[ManifestEntry] = []
    skipped: list[SkippedEntry] = []
    seen: set[str] = set()
    for i, item in enumerate(raw):
        doc_id = item.get("doc_id") if isinstance(item, dict) else None
        try:
            entry = ManifestEntry.model_validate(item)
        except ValidationError as exc:
            fields = ", ".join(".".join(map(str, e["loc"])) or "entry" for e in exc.errors())
            skipped.append(SkippedEntry(i, doc_id, f"invalid fields: {fields}"))
            continue
        if entry.doc_id in seen:
            skipped.append(SkippedEntry(i, entry.doc_id, "duplicate doc_id"))
            continue
        if not entry.file.lower().endswith(SUPPORTED_EXTENSIONS):
            skipped.append(SkippedEntry(i, entry.doc_id, "unsupported file type"))
            continue
        if not os.path.isfile(os.path.join(corpus_dir, entry.file)):
            skipped.append(SkippedEntry(i, entry.doc_id, "file not found in corpus"))
            continue
        seen.add(entry.doc_id)
        entries.append(entry)

    for s in skipped:
        log.warning("manifest entry skipped", extra={"index": s.index, "doc_id": s.doc_id, "reason": s.reason})
    return entries, skipped
