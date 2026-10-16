import json
import sqlite3
import time
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.data.db import connect, init_app_tables
from app.main import create_app
from tests.fixtures.fixture_db import build_fixture_db


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings pointing at a fresh synthetic DB and a tiny manifest; never reads .env."""
    db_path = tmp_path / "veltrix.db"
    build_fixture_db(str(db_path))
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([]), encoding="utf-8")
    return Settings(
        _env_file=None,
        db_path=str(db_path),
        corpus_dir=str(corpus),
        manifest_path=str(manifest),
        cache_dir=str(tmp_path / "cache"),
        app_today="2026-09-15",
    )


def wait_ready(client: TestClient, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client.get("/health").status_code == 200:
            return
        time.sleep(0.02)
    raise AssertionError("service did not become ready")


@pytest.fixture
def db_path(tmp_path) -> str:
    """A fresh synthetic DB with the app tables added."""
    path = str(tmp_path / "repo.db")
    build_fixture_db(path)
    init_app_tables(path)
    return path


@pytest.fixture
def db_conn(db_path) -> Iterator[sqlite3.Connection]:
    with connect(db_path) as conn:
        yield conn


@pytest.fixture
def client(settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as c:
        wait_ready(c)
        yield c


class HashingEmbedder:
    """Deterministic stand-in for the local embedding model: hashed bag of words,
    unit length. Keeps tests fast and offline; shared words → similar vectors."""

    model_name = "test-hashing"

    def embed(self, texts):
        import re
        import zlib

        import numpy as np

        out = np.zeros((len(texts), 384), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                out[row, zlib.crc32(word.encode()) % 384] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)

    def embed_query(self, query):
        return self.embed([query])[0]


@pytest.fixture(autouse=True)
def _no_model_downloads(monkeypatch):
    """Tests never load the real embedding model (network, seconds per test)."""
    monkeypatch.setattr("app.ingestion.pipeline.Embedder", HashingEmbedder)
    monkeypatch.setattr("app.main.Embedder", HashingEmbedder)
