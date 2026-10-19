# DESIGN — Veltrix PeopleOps Assistant

**Approach:** metadata-filtered hybrid RAG with facts extracted at ingestion. Eligibility, precedence, permissions, leave
rules and date arithmetic are deterministic code; the LLM plans the request (no corpus text) and writes the answer. No
agent framework, so access filtering, the token budget and precedence stay explicit, testable code
(reasoning: `docs/DECISIONS_QA.md`).

## 1. Ingesting tables and the scanned document
- Loaders (PDF `pdfplumber`, DOCX, HTML, Markdown) emit one ordered block list (heading / text / table) (`ingestion/blocks.py`);
  everything downstream is format-agnostic. Each document's metadata table is dropped; the manifest is authoritative.
- **Tables** stay whole as markdown in chunks; one over the size limit splits by row groups with the header repeated.
  Holiday calendars become typed rows (`extract_holidays.py` → `holidays`), with "restricted" read from section headings.
- **Scanned pages** (no text layer) are rendered and transcribed by a vision LLM with a verbatim-only, tool-less prompt
  (`loaders/ocr.py`). Numbers are cross-checked against Tesseract and mismatches flagged low-confidence; Tesseract is the
  fallback. Results are cached by file and engine setup. Nothing is typed in by hand.

## 2. Chunking and retrieval
- **Section chunks** (~350 tokens, max 500) keep a rule with its conditions and give each citation a heading locator; the
  heading path is prefixed for retrieval (and counted as corpus tokens). Winner, conflicting and upcoming passages fit
  together under 3,000 tokens; fixed-size chunks split rules and lose locators.
- **Hybrid search** over **only the eligible chunk ids**: BM25 (exact terms: holiday names, codes) and exact cosine search
  over local `bge-small` vectors in FAISS `IndexFlatIP` with an `IDSelectorBatch` (paraphrases), merged by weighted RRF
  (k = 60, BM25 at 0.5: at equal weight it pushed bereavement passages out of the top 10). A passage naming another
  document by title brings that document in (`retriever.py`).
- **Budget:** `retrieval/context_packer.py::pack_context` fills to 3,000 `cl100k_base` tokens (in search-rank order, each hit preceded by
  the winning, conflicting and upcoming passages of the rules it raised); `llm/client.py::LLMClient.complete` recounts and refuses anything over, and logs `corpus_tokens`.

## 3. Implementing the Section 5 rules
- **Facts:** at ingestion an LLM extracts `{topic, attribute, value, unit, conditions, quote}` per chunk
  (`extract_facts.py`; quotes must exist in the source; flagged chunks are skipped).
- **Same rule:** fact keys are embedded and clustered (average linkage, cosine 0.12) into a `rule_key`, with cannot-links:
  same document, different numbers in the key, recommendation vs entitlement (`rule_keys.py`). Pairs just outside the
  threshold go to an LLM judge, "same rule?" (`rule_judge.py`), cached per question. No hand-written taxonomy.
- **Eligibility** (`retrieval/eligibility.py::is_eligible`): readable access level, region or `GLOBAL`, effective date ≤
  the reference date (`APP_TODAY`, or a past date asked about). **Precedence** (`retrieval/precedence.py::resolve`): per
  rule, authority → scope → recency; a full tie with different values is a conflict and both are cited; a future document
  is mentioned only if it would win on its effective date. Nothing eligible → `no_answer`.
- The composer gets a rule table (current / outranked / upcoming per rule); `validator.py::superseded_values` sends an
  answer stating an outranked value back for one rewrite.

## 4. Where access control is enforced
| Guarantee | File :: function |
|---|---|
| Identity, role and region from the DB only; missing or unknown header → 401 | `main.py::_requester` (in `create_app`) |
| A session belongs to its first sender → 403 | `data/sessions.py::claim_session` |
| Restricted documents never searched or sent | `retrieval/eligibility.py::allowed_chunk_ids` → `retrieval/index.py::search(allowed_ids=…)` |
| Leave-data permissions inside every tool | `tools/authz.py::can_view_employee`, `can_view_team`, `owns_request`, via `tools/registry.py` |
| Denied and non-existent employees get the same refusal | `tools/authz.py::ACCESS_DENIED` |
| Bulk, manipulation, more than one other employee | `orchestration/orchestrator.py::_guard` |
| Citations only from chunks sent and allowed | `orchestration/validator.py::build_citations` |

Tools get the requester from the header, never from LLM arguments. The planner never sees corpus text, so a document can't
trigger a tool. Writes are server-side proposals executed only after a clear "yes", in a `BEGIN IMMEDIATE` transaction
(`data/db.py`) that re-checks every rule, so concurrent requests can't overdraw or refund twice.

## 5. Keeping personal data out of logs and prompts
- Repositories never select `phone`; emails leave the data layer masked (`privacy/masking.py`).
- The planner sees the requester's id, role and region only; the composer gets names only for employees the requester may
  see. `validator.py::scrub_answer` removes phones, masks emails and hides names the requester may not see.
- `privacy/log_filter.py::PiiLogFilter` redacts emails, phones and known names in every log record. LLM repair messages
  name the invalid fields, never the rejected values (which can quote corpus text).
- Memory is the current session only; every turn reloads identity and re-runs filtered retrieval.

## 6. Failure modes (found by `make eval`, 53 cases × 3 runs; `eval/HISTORY.md`)
| Failure | Status |
|---|---|
| Outdated v1 or FAQ value stated as current | Fixed: authority ranking, superseded-value rewrite. Rare v1 citation left (split rule) |
| Same rule not recognised (India "no fixed expiry" vs global "31 March"; scanned WFH memo vs policy) | Mostly fixed: conditions in the key, LLM judge for near misses |
| Judge batches lost to malformed JSON (`"same":false]}`), so rules stayed split | Fixed: flat reply format, verdicts read pair by pair |
| Different rules merged (secondary caregiver "two blocks" / "4 weeks") | Unfixed; the composer sees each source's own label and quote |
| Wiki's planted instruction answered ("salary band… not provided") | Fixed: instruction-like sentences of flagged chunks never reach the composer; prompt forbids mentioning them |
| Future-balance arithmetic wrong (10, 12, 33 days) | Fixed: credits counted in code (`future_balance.py`) |
| Planner output: function-call shape, `"modify"` as action, `"many_employees"` as intent, a question planned as a booking | Fixed: explicit shapes, repair says what was wrong, code corrects the known patterns |
| Provider stalls (calls hanging 12–15 s) | Mitigated: a second identical call at 5 s (planner) / 7 s (composer); measured maxima 4.3 / 4.9 s |
| Ingestion too slow for 10 minutes | Fixed: batching, concurrency, low reasoning effort → 240 s cold in Docker |
| Small-model variance | Open: 155–157/159 case-runs correct in the last three evals; a different 2–4 fail once each time |

## 7. What breaks first at 10,000 documents
1. **Ingestion time:** per-chunk LLM extraction won't fit 10 minutes → offline, incremental ingestion keyed by content hash.
2. **Rule grouping:** clustering and judge pairs grow O(n²) → ANN-based incremental assignment, plus a document lineage
   link ("replaces v1") so versions are compared without similarity guesses.
3. **Context budget:** more versions per rule → resolve precedence on facts first and send only winning and conflicting passages.
4. **Filtered search:** move `index.py` to Qdrant or pgvector with payload indexes on access level, region and date.
5. **Writes and instances:** SQLite's single writer → Postgres with row-level locking.
