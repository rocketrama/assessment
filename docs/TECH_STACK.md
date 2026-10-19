# Tech Stack — Veltrix PeopleOps Assistant

> **Principle:** the smallest stack that meets the spec, with a defensible reason for every choice. Security-critical logic (access filtering, token budget, precedence) lives in our own readable code, not hidden inside framework defaults.

---

## 1. Core stack

| Layer | Choice | Why (spec reference) |
|---|---|---|
| **Language** | Python 3.11 | §1 requires 3.10+; 3.11 is faster and stable. |
| **API** | FastAPI + Uvicorn | The automated tests depend on the exact contract (§9); Pydantic validation gives 400 handling. |
| **Schemas** | Pydantic v2 | Strict request/response models; validates LLM JSON output. |
| **LLM** | Anthropic SDK and the OpenAI SDK behind one `Provider` interface (`llm/client.py`); the OpenAI client also serves Gemini, NVIDIA, OpenRouter and Mistral through their OpenAI-compatible endpoints | §10.3: at least one of OpenAI, Anthropic or Gemini; reviewers use their own key. Eval runs used `openai/gpt-oss-20b` on NVIDIA. |
| **Token counting** | `tiktoken` (`cl100k_base`) | Required by §4.3 (3,000 corpus-token cap). Encoding pre-cached at Docker build time (`TIKTOKEN_CACHE_DIR`) — tiktoken otherwise downloads it on first use. |
| **Embeddings** | `fastembed` — `BAAI/bge-small-en-v1.5` (CPU, ONNX) | §10.3: local embeddings must run on CPU. No PyTorch → small image, fast startup. |
| **Vector store** | FAISS `IndexFlatIP` (`faiss-cpu`), exact inner-product search with an `IDSelectorBatch` of the allowed chunks (`retrieval/index.py`) | Local (§10.3) and named in the spec. Exact search, built in memory at startup from vectors stored in SQLite, so there is no index file to go stale. It started as a numpy dot product; on all 56 eval questions FAISS returns the identical top 10. |
| **Keyword search** | BM25 with Lucene IDF, ~20 lines | Lexical half of hybrid retrieval. `rank_bm25`'s Okapi IDF is 0 for a term in half the documents — common in small allowed sets. |
| **Fusion** | Reciprocal Rank Fusion (own code) | Few parameters, stable, deterministic. |
| **Rule-key clustering** | Average linkage, cosine distance, distance threshold, same-document cannot-link — ~25 lines of numpy (`ingestion/rule_keys.py`) | Unknown number of rules, many 2-fact groups, no chaining → agglomerative. Planned with scipy; Windows Application Control blocked scipy's DLLs on the dev machine, so the algorithm is implemented and unit-tested in numpy (one dependency fewer). |
| **Rule judge** | The same LLM, asked "same rule?" for cluster pairs just outside the threshold (`ingestion/rule_judge.py`) | Embeddings can't tell "remote work days" from "hybrid work days" reliably at a safe threshold; a cached yes/no per pair can, and cannot-links still forbid unsafe merges. |
| **Cache storage** | Separate SQLite `cache.db` (outside `./data`) | Disposable derived data; transactional (atomic writes); never committed. A design choice — §2.3 would also allow these tables in `veltrix.db`. |
| **Database** | SQLite via standard-library `sqlite3` | `veltrix.db` is given; add tables only (§2.3); built-in transactions. |
| **PDF** | `pdfplumber` (text + tables), `pypdfium2` (render pages for OCR) | Tables are required (§4.1). |
| **DOCX** | `python-docx` | Headings and tables. |
| **HTML** | `beautifulsoup4` + `lxml` | Headings and `<table>`. |
| **Markdown** | `markdown-it-py` | Headings and pipe tables. |
| **OCR** | Vision LLM first; `pytesseract` + `tesseract-ocr` fallback | §4.1 allows either; the fallback keeps ingestion alive. |
| **Dates** | Standard-library `datetime` + `python-dateutil` | Calendar dates only, no time zones (§8.4). |

---

## 2. Quality and delivery

| Area | Choice |
|---|---|
| **Tests** | `pytest`, `pytest-asyncio`, `httpx` (FastAPI test client), `hypothesis` (token-budget property test) |
| **Evaluation** | Own harness `eval/run_eval.py` + YAML test cases → `make eval` scorecard |
| **Lint / format** | `ruff` |
| **Type checking** | `mypy` |
| **Logging** | Standard `logging`, JSON output, custom PII redaction filter |
| **Container** | `Dockerfile` (`python:3.11-slim` + `tesseract-ocr`; embedding model and tiktoken encoding downloaded at build time) |
| **Orchestration** | `docker-compose.yml` — one service, port `8000`, mounts `./data` |
| **Task runner** | `Makefile`: `make test`, `make eval`, `make lint`, `make run` |
| **Dependencies** | Pinned `requirements.txt` (or `uv` lockfile) |

---

## 3. Deliberately not used

| Skipped | Reason |
|---|---|
| **LangChain / LlamaIndex** | Allowed (FAQ §13), but they hide exactly what reviewers assess: filtering before retrieval, the token budget and precedence. ~200 lines of our own retrieval code are easier to audit and defend. |
| **Chroma / Qdrant / pgvector** | Fine at scale, but extra overhead for 30 documents. Named as the 10,000-document upgrade path in DESIGN.md. |
| **scikit-learn / scipy** | Only needed for clustering, which is a few lines of numpy here (scipy was also blocked by Windows Application Control on the dev machine). |
| **PyTorch / sentence-transformers** | Heavy image and slow build; fastembed does the same on CPU. |
| **Redis / Postgres** | §10.3 allows only LLM and embedding APIs as external services. SQLite is enough. |
| **GraphRAG** | The hard problem is versioning and precedence, not entity relationships. Corpus-derived summaries count toward the 3,000-token budget and risk mixing access levels. |
| **UI (Streamlit, etc.)** | Out of scope (§3). Time goes to tests and evaluation instead. |

---

## 4. Scale-up path (10,000 documents)

| Today | At scale |
|---|---|
| FAISS exact search + ID allow-list | FAISS IVF/HNSW, or Qdrant / pgvector with payload indexes on `access_level`, `region`, `effective_date` |
| Ingestion at startup | Offline incremental ingestion with content-hash cache |
| Pairwise near-duplicate check | MinHash LSH |
| Fact-key clustering O(n²) | Approximate-nearest-neighbour clustering, incremental assignment |
| SQLite | Postgres with row-level locking for concurrent leave writes |
| Precedence by comparing facts | Lightweight document-lineage graph (supersedes / amends / contradicts) |

---

## 5. One-line summary for DESIGN.md

> Deliberately minimal stack: FastAPI + SQLite + BM25/dense (FAISS) + fastembed, no orchestration framework — so access filtering, the token budget and precedence are explicit, auditable code rather than framework defaults.
