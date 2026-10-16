"""Create a synthetic data/veltrix.db for local runs (fake people only).

Usage: python scripts/make_dev_db.py [path]   (default: ./data/veltrix.db)
Refuses to overwrite an existing file, so a real database is never replaced.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.fixtures.fixture_db import build_fixture_db  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "veltrix.db")
if os.path.exists(path):
    sys.exit(f"{path} already exists; not overwriting")
build_fixture_db(path)
print(f"created synthetic database: {path}")
