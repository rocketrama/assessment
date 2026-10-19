PYTHON ?= python

.PHONY: install test run up dev-db eval ragas

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

# RAGAS scores (faithfulness, answer relevancy, context precision/recall) for eval/results.json;
# needs its own venv: python -m venv .venv-eval && .venv-eval/Scripts/python -m pip install -r requirements-eval.txt
EVAL_PYTHON ?= .venv-eval/Scripts/python
ragas:
	$(EVAL_PYTHON) eval/ragas_eval.py
