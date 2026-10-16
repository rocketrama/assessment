import json
import time
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
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
def client(settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as c:
        wait_ready(c)
        yield c
