PYTHON ?= python

.PHONY: install test run up dev-db eval

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

test:
	$(PYTHON) -m pytest

run:
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000

up:
	docker compose up --build

# Synthetic data/veltrix.db (fake people) for local runs; never overwrites an existing file
dev-db:
	$(PYTHON) scripts/make_dev_db.py

# Scored question set against the real model, 3 runs (needs the LLM key in .env)
eval:
	$(PYTHON) eval/run_eval.py
