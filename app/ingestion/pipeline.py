"""Ingestion entry point. Step 1: validates inputs only; parsing, OCR, chunking and
indexing are added in step 4. `/health` stays 503 until this returns successfully."""

import json
import os

from app.config import Settings


class IngestionError(Exception):
    pass


def run_ingestion(settings: Settings) -> None:
    if not os.path.isfile(settings.manifest_path):
        raise IngestionError(f"manifest not found: {settings.manifest_path}")
    if not os.path.isdir(settings.corpus_dir):
        raise IngestionError(f"corpus dir not found: {settings.corpus_dir}")
    with open(settings.manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, list):
        raise IngestionError("manifest.json must be a JSON array")
