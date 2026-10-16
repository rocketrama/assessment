# DESIGN — Veltrix PeopleOps Assistant

> Draft v2, written before building. Every file/function named here must exist in the final code; §6 must contain real findings. Detailed reasoning: `docs/DECISIONS_QA.md`.

**Approach:** metadata-filtered hybrid RAG with ingestion-time structured extraction. Eligibility, precedence, permissions and leave rules are deterministic code; the LLM only interprets the user and writes the answer. Deliberately minimal stack — FastAPI, Pydantic, SQLite, FAISS + BM25, fastembed, tiktoken, no orchestration framework — so access filtering, the token budget and precedence are explicit, auditable code.

## 1. Ingesting tables and the scanned document
- Each loader (PDF `pdfplumber`, DOCX `python-docx`, HTML `beautifulsoup4`, Markdown `markdown-it-py`) emits one block model `{doc_id, kind, heading_path, page, text, rows}`; downstream code is format-agnostic. `manifest.json` is validated with Pydantic; bad entries are logged.
- **Tables** stay whole as markdown inside chunks (for the LLM) and are also stored as typed rows: calendar tables → `holidays(doc_id, region, date, name)`; accrual tables → policy facts. Large tables split by row groups, header repeated.
- **Scanned pages** (detected per page by an empty text layer) are rendered at ~300 DPI and transcribed by a vision LLM with a verbatim-only, **tool-less** prompt, producing markdown with headings and tables. Numbers and dates are **cross-checked against Tesseract**; mismatches are flagged low-confidence. Tesseract is the fallback if the vision call fails. Output is cached by file hash and treated as untrusted corpus text.
- Nothing from the corpus is hardcoded; extracted values are validated (dates parse, quoted spans exist in the source).

## 2. Chunking and retrieval
- **Section-based chunks** (~350 tokens, max 500): each chunk's locator is its nearest heading (§4.2), rules stay with their conditions, and winner + conflicting + upcoming versions fit together in 3,000 tokens. No headings → page chunks with `p. N`. A short heading path is prefixed to improve retrieval (counted as corpus tokens).
- **Hybrid retrieval:** BM25 (`rank_bm25`) catches exact terms (holiday names, codes); FAISS exact `IndexFlatIP` over local `bge-small` embeddings catches paraphrases. **Both search only the eligible chunk IDs.** Results merge with **RRF, k = 60**: rank-based (no score calibration), no α to overfit on a small eval set, stable on the modified corpus, deterministic (ties by `chunk_id`). Top 8–10 go forward, so single-retriever exact hits survive. **Near-duplicates** are collapsed to one eligible representative (chosen by precedence) **only when their extracted facts are identical** — text-similar versions with different values both reach the precedence resolver, so conflicts are never hidden.
- **Why these components:** FAISS stores only vectors and IDs — access metadata lives only in SQLite (one source of truth, filter in our tested code, not a vector-DB `where` clause). Local fastembed means ingestion needs only the one LLM key reviewers supply (Anthropic has no embeddings API), with deterministic vectors and no network latency.
- **Budget:** corpus text travels as typed `CorpusSegment`s. `retrieval/context_packer.py` fills to 3,000 `cl100k_base` tokens (winning facts → conflicts → upcoming → other) and `llm/client.py::LLMClient.complete` recounts and refuses any call above 3,000; `corpus_tokens` is logged on every call, ingestion included. The tiktoken encoding is pre-cached at Docker build time.
- **Rejected:** naive RAG (can't apply precedence), agentic RAG (unbounded latency, variable output across 3 runs), GraphRAG (summaries consume the budget and can mix access levels), fixed-size/semantic chunking (unclear locators, split rules), weighted fusion, Chroma.

## 3. Implementing the Section 5 rules
- **Rule extraction:** at ingestion, an LLM extracts facts per chunk `{topic, attribute, value, unit, conditions}`; fact keys are embedded and clustered into a canonical `rule_key` — how "the same rule" is recognised without a hand-written taxonomy.
- **Eligibility** (`retrieval/eligibility.py::is_eligible`): readable `access_level` ∧ region or `GLOBAL` ∧ `effective_date ≤ reference_date` (`APP_TODAY`, or the past date asked about).
- **Precedence** (`retrieval/precedence.py::resolve`): per `rule_key`, sort by authority → scope → recency; full tie with different values → conflict, both cited.
- **Upcoming change:** mentioned only if `resolve` at its effective date makes it the winner. **No eligible answer** → `no_answer`; uncovered parts stated.
- The spec's worked example is a unit-test fixture.

## 4. Where access control is enforced
| Guarantee | File :: function |
|---|---|
| Identity, role, region from DB only; missing/unknown header → 401 | `app/main.py::resolve_requester` |
| Session ownership → 403 | `data/sessions.py::assert_owner` |
| Restricted docs never searched or sent | `retrieval/eligibility.py::allowed_chunk_ids` → `retrieval/index.py::search(allowed_ids=…)` |
| Leave-data permissions | `tools/authz.py::can_view_employee`, `can_view_team`, `assert_self`, called inside every tool in `tools/registry.py` |
| Same refusal for denied and non-existent employees | `tools/authz.py` → single `ACCESS_DENIED` |
| Bulk requests refused | `orchestration/orchestrator.py::guard_plan` |
| Citations only from used ∩ allowed chunks | `orchestration/validator.py::build_citations` |

Tools get `ToolContext(requester)` from the header, never identity from LLM args. LLM output and tool args are Pydantic-validated in strict mode; tool schemas sent to the LLM are generated from the same models. The planner never sees corpus text, so documents cannot trigger tools. Writes use a server-side proposal; only stored args execute on confirmation. Apply/cancel run in **`BEGIN IMMEDIATE`** transactions (WAL, busy timeout, `isolation_level=None`), re-validating every rule inside the lock so concurrent requests can't produce negative balances, overlaps or double refunds.

**Assumption — `./data` is writable.** Sessions, messages and proposals are new tables in `veltrix.db` (not a separate file) so a proposal is claimed in the same transaction as the leave write it executes. Cost: the reviewers' DB gains three tables and is switched to WAL (`-wal`/`-shm` files appear beside it). Existing table schemas are never altered. Rejected: a separate `app.db`, which would need a cross-database transaction for confirm-and-apply.

## 5. Keeping personal data out of logs and prompts
- Repositories never select `phone`; emails leave the data layer masked (`r***@domain`); `tool_calls` args are serialized through a masking model.
- The planner sees employee IDs, role and region only; the composer gets names only for employees the requester may see, when needed.
- `orchestration/validator.py` strips phones, masks emails and checks names before responding; `privacy/log_filter.py` redacts email, phone and known names in all logs.
- **Memory** is per session only (owner, last few turns as answer text, ≤ 1 proposal); every turn reloads identity and re-runs filtered retrieval. Long-term memory was rejected: the header is trusted but not authenticated, and it would break reproducibility across test runs (§10.3) and §9.1. No answer or tool-result caching; only provider prompt caching of the static system prompt.

## 6. Failure modes
| Failure mode | Status | Fix / note |
|---|---|---|
| Outdated FAQ overrides policy | _to verify_ | Authority ranking in `precedence.resolve` |
| Same rule phrased differently not detected as a conflict | _to verify_ | Fact-key clustering threshold |
| Vision OCR invents or "corrects" numbers | _to verify_ | Tesseract number cross-check, low-confidence flag |
| PDF headings not detected → weak locators | _to verify_ | Font heuristics, `p. N` fallback |
| Document text tries to trigger a tool | _to verify_ | Planner never sees corpus; validator blocks actions |
| FastAPI returns 422 instead of 400/401 | _to verify_ | Custom handlers; contract tests |
| Concurrent apply drives balance negative | _to verify_ | `BEGIN IMMEDIATE`; two-thread test |
| Malformed LLM JSON | _to verify_ | Strict validation → one repair retry → safe fallback |

_Replace with real findings, including unfixed ones._

## 7. What breaks first at 10,000 documents
1. **Ingestion time** — per-chunk LLM extraction can't fit 10 minutes → offline incremental ingestion with a content-hash cache.
2. **Context budget** — more versions per rule → resolve precedence on facts before retrieval; send only winning and conflicting passages.
3. **Fact clustering** O(n²) → ANN clustering with incremental assignment.
4. **Filtered vector search** → Qdrant or pgvector with payload indexes on access level, region, date; MinHash LSH for near-duplicates.
5. **Precedence lookups** → a lightweight document-lineage graph (supersedes / amends / contradicts) in SQL — not GraphRAG.
6. **Concurrent writes / multiple instances** → Postgres with row-level locking.
