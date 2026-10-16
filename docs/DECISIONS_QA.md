# Design Decisions — Questions & Answers

> Record of the key questions raised while planning the Veltrix PeopleOps Assistant (Senior AI Engineer assessment) and the reasoning behind each answer. Spec section numbers refer to `TASK.pdf`.

**Contents**
- **Part 1 — Retrieval, stack and architecture:** Q1–Q25
- **Part 2 — Security, privacy and access control:** Q26–Q31
- **Part 3 — Orchestration and LLM behaviour:** Q32–Q39, Q48
- **Part 4 — Data, ingestion and leave rules:** Q40–Q44
- **Part 5 — Testing, evaluation and delivery:** Q45–Q47
- **Part 6 — Production readiness:** Q50
- **Summary lines for DESIGN.md:** Q49
- **Assumptions to record in the README:** end of file

---

# Part 1 — Retrieval, stack and architecture

## Q1. What does the assessment expect from a senior AI engineer?

**Answer:** The required features are only the baseline. Seniority shows in how the hard, ambiguous parts are handled and how clearly the trade-offs are explained.

- **Must-pass first (§12.1):** `docker compose up` + `/health` within 10 min; no restricted document ever sent to the LLM; no unauthorised leave data; ≤ 3,000 corpus tokens per LLM call; works on a modified corpus.
- **Deterministic logic in code:** precedence, eligibility, working days, permissions and leave rules are testable functions; the LLM only interprets and writes.
- **Rule extraction at ingestion** so "the same rule" conflicts (§5) are detected in code.
- **Eval harness** (`make eval`) with route accuracy, citation precision/recall, refusal correctness, run 3× for consistency.
- **Defence in depth** for security and prompt injection, with adversarial tests.
- **Reliability:** malformed LLM output, timeouts within 30 s, partial ingestion failure.
- **Honest DESIGN.md:** trade-offs, real failure modes (fixed and not fixed), concrete 10,000-document analysis.
- **Clean commit history** — reviewers may read it (§11.2).

---

## Q2. Which type of RAG do we use?

**Answer:** **Filtered hybrid RAG with structured extraction.** Not naive RAG, not agentic RAG, not GraphRAG.

| Layer | What | Why |
|---|---|---|
| 1. Pre-filtered retrieval | Filter by role, region, effective date **before** search | §7.2: removing restricted docs after the LLM sees them "is not enough"; must-pass #2 |
| 2. Hybrid search | BM25 + embeddings, fused with RRF | HR questions mix exact terms and paraphrases; neither method alone covers both |
| 3. Structured extraction | Holidays and rule values into tables at ingestion | Working days need exact holidays (§6.2); precedence needs "the same rule" (§5); structured records don't count toward the 3,000 tokens (§4.3) |
| 4. Code decides, LLM writes | Eligibility, precedence and conflicts in code | Testable, reproducible, safe |

**Rejected alternatives:**
- **Naive RAG** — cannot apply authority/scope/recency; fooled by outdated FAQs.
- **Agentic RAG** — unpredictable latency (30 s limit) and inconsistent results across 3 test runs; lets the LLM steer retrieval.
- **GraphRAG** — 30 documents don't need a knowledge graph (see Q3).

---

## Q3. Why not GraphRAG — even if we grow to 10,000 documents?

**Answer:** Full GraphRAG still doesn't fit, but a **lightweight document-lineage graph** would.

**Why GraphRAG doesn't fit:**
1. **Wrong problem** — the hard part is *which version of a rule applies* to this person on this date (versioning + metadata), not entity relationships.
2. **Breaks the token budget** — community summaries are "summaries made from the corpus" and count toward the 3,000 tokens (§4.3).
3. **Access-control risk** — summaries mixing `hr` and `all` documents leak restricted content (must-pass #2).
4. **Cost and freshness** — LLM indexing of 10,000 docs is expensive; every policy change forces re-summarisation.

**What breaks first at 10,000 docs and the fix:**

| Breaks first | Fix |
|---|---|
| Ingestion time | Offline incremental ingestion with content-hash cache |
| Too many versions per rule for 3,000 tokens | Resolve precedence on facts before retrieval; send only winning + conflicting passages |
| Fact-key clustering is O(n²) | Approximate-nearest-neighbour clustering, incremental assignment |
| Flat FAISS + ID allow-lists | Qdrant / pgvector with payload indexes |
| Near-duplicate detection | MinHash LSH |

**Senior move:** a document-lineage graph (`supersedes`, `amends`, `contradicts`) built from metadata and facts, stored in SQL tables — makes precedence scale without LLM summaries, and each node keeps its `access_level`.

**Note:** build for 30 documents now (§1: correctness and depth over breadth). The 10,000-document answer belongs in DESIGN.md (§11.1), not in code.

---

## Q4. Which tech stack, and why?

**Answer:** The smallest stack that meets the spec, with a reason for every choice. Full detail in `TECH_STACK.md`.

| Layer | Choice | Reason |
|---|---|---|
| API | FastAPI + Pydantic v2 | Exact contract (§9), built-in validation for 400 |
| LLM | Anthropic default, OpenAI adapter, own `LLMProvider` interface | §10.3; reviewers use their own key |
| Tokens | `tiktoken` `cl100k_base` | Required by §4.3 |
| Embeddings | `fastembed` `bge-small-en-v1.5` (CPU) | §10.3 local CPU; no PyTorch |
| Vector store | FAISS | Local; exact search; ID pre-filtering |
| Keyword search | `rank_bm25` | Hybrid retrieval |
| Database | SQLite | Given; add tables only (§2.3) |
| Parsing | `pdfplumber`, `pypdfium2`, `python-docx`, `beautifulsoup4`, `markdown-it-py` | All formats + tables (§4.1) |
| OCR | Vision LLM + Tesseract fallback | §4.1 allows either |
| Quality | pytest, hypothesis, ruff, mypy, Makefile, Docker | §10, §11 |

**Not used:** LangChain/LlamaIndex (hide filtering/budget/precedence), Chroma/Qdrant (overhead for 30 docs), PyTorch (heavy image), Redis/Postgres (§10.3 allows only LLM and embedding APIs as external services), UI (out of scope §3).

---

## Q5. Why not LangGraph for orchestration?

**Answer:** LangGraph is allowed (FAQ §13) and not wrong — but our flow doesn't need what it is good at (cycles, multi-agent routing, checkpointing, long-running workflows).

| Our requirement | Why plain Python fits better |
|---|---|
| Fixed pipeline: plan → guard → tools/retrieval → compose → validate | Linear; a graph adds structure without value |
| ≤ 2 LLM calls, < 30 s (§9.1) | Open-ended agent loops make latency hard to bound |
| Consistent output across 3 runs (§10.3) | Fewer LLM decisions about flow = more reproducible |
| Visible security (§6.1, §7.2) | Every check readable top to bottom in `orchestrator.py`, not spread across nodes and edges |
| Propose/confirm (§8.2) | A 4-state machine + one `proposals` table; LangGraph interrupts would add a checkpointer and hidden state |
| Sessions already in SQLite | Avoids a second source of truth |
| Docker reliability (must-pass #1) | Fewer dependencies, fewer failure points |

**When LangGraph would be the right choice:** multi-step tool reasoning in loops, multiple agents, or long-running approval workflows (out of scope, §3).

**If used anyway:** keep permission checks inside tool functions, keep filtering inside the retriever, cap recursion, and keep SQLite as the single source of session truth.

---

## Q6. What memory does the RAG use?

**Answer:** Two kinds — **knowledge memory** (what retrieval reads from) and **conversation memory** (what the chat remembers).

**Knowledge memory** — built once at ingestion, stored locally:

| Store | Holds | Used for |
|---|---|---|
| FAISS index | Embeddings of every chunk | Semantic search |
| BM25 index (RAM, rebuilt at startup) | Tokenized chunk text | Keyword search |
| SQLite `chunks` | Text + `doc_id`, locator, `access_level`, regions, `effective_date` | Pre-filtering, citations, context packing |
| SQLite `policy_facts` | Typed rules (rule key, value, unit, source chunk) | Precedence and conflicts (§5) |
| SQLite `holidays` | Dates per region and document | Working days (§6.2) |
| SQLite `documents` | Manifest metadata + content hash | Eligibility, ingestion cache |

- Metadata lives with every chunk, so the access filter runs **before** search (must-pass #2).
- Structured records don't count toward the 3,000 tokens (§4.3); chunk text does.
- Stored in a cache directory **outside `./data`**, because reviewers replace `./data` (§10.1).

**Conversation memory** — short-term, per session, in SQLite:

| Table | Holds | Why |
|---|---|---|
| `sessions` | `session_id` → owning `employee_id` | §7.1: different employee on same session → 403 |
| `messages` | Last N turns | Follow-ups such as "make it until Friday instead" (§8.2) |
| `proposals` | At most one pending write | Propose/confirm (§8.2) |

- Send the planner only the last 4–6 turns.
- Never store raw corpus chunks in history — each turn re-runs filtered retrieval.
- No personal data in history beyond what that user was already shown.
- Proposal args come from the `proposals` table, never from the LLM re-reading history.

**Not built:** long-term cross-session memory (privacy risk, no score benefit) and semantic answer caching (stale when reference date or region changes).

---

## Q7. Which cache is used — KV cache?

**Answer:** No. A **KV cache** is internal to transformer inference (stored attention keys/values); with a hosted API (Anthropic/OpenAI) the provider manages it. We'd only control it when self-hosting (e.g. vLLM).

**Closest equivalent we use — provider prompt caching:**
- Anthropic: explicit `cache_control` breakpoints; OpenAI: automatic prefix caching.
- **Cache:** system prompt + tool schemas (identical on every call → lower latency and cost, helps the 30 s limit §9.1).
- **Don't cache:** retrieved chunks (change per question).
- Caching does **not** change the 3,000-token rule — corpus text in a call counts whether cached or not.
- Safe because the system prompt contains no corpus content (§4.3).

**Caches the project actually uses:**

| Cache | What | Why |
|---|---|---|
| Ingestion cache | Blocks, chunks, embeddings, facts keyed by `sha256(file) + manifest entry + pipeline version` | Fast restarts; unchanged docs skip OCR and LLM extraction → under the 10-min `/health` limit; key at 10,000 docs |
| Model cache | Embedding model downloaded at Docker build time | No network download at startup (must-pass #1) |
| Prompt cache | System prompt + tool schemas at the provider | Latency and cost |
| In-memory indexes | FAISS + BM25 loaded once | Retrieval < 100 ms |

**The content-hash ingestion cache in detail:**

*Why cache:* OCR, per-chunk LLM extraction and embeddings are the slowest part; without a cache every restart, eval run and dev change repeats them — slow, costly, and adds extraction variance (Q35).

*Why a content hash, not file name or mtime (critical):* reviewers replace `./data` with a **modified corpus** (§10.1, must-pass #5) — **file names stay the same, content changes** (12 days → 14). A name-keyed cache serves **old facts** → answers show old values → silent must-pass #5 failure. Modification times are unreliable when files are copied. **sha256 of the file bytes** changes whenever content changes.

*Cache key:*

| Part | Why |
|---|---|
| sha256 of file bytes | Content changed → re-process |
| Pipeline version | Parser/chunker code changed → re-process |
| Model IDs (vision, extraction, embedding) | Different model → different output |
| Prompt version (OCR, extraction) | Prompt change → different output |

*Security point — metadata is never cached:* `access_level`, `regions`, `effective_date`, `doc_type` come from `manifest.json` and drive access control and precedence. If the manifest changes a doc from `all` to `hr`, a cached `all` would **leak restricted content** (must-pass #2). → The cache stores only **expensive derived content** (OCR text, chunks, facts, embeddings); **manifest metadata is read fresh at every startup** and attached then. Documents **not in the manifest** are dropped even if cached — the manifest is the source of truth.

*Per-stage caching:* parse/OCR → chunks → facts → embeddings, each cached separately (a chunker change re-runs only later stages; a new `RULE_CLUSTER_THRESHOLD` re-runs only clustering). **Rule-key clustering is always recomputed** — it depends on all facts together and takes seconds.

*Why per stage — each stage has a different cost and different reasons to re-run:*

| Stage | Cost | Re-runs when… |
|---|---|---|
| 1. Parse / OCR | High for scans (vision LLM, seconds per page) | File bytes, OCR model or prompt change |
| 2. Chunks | Cheap | Stage-1 output or chunker code changes |
| 3. Facts | **Highest** (one LLM call per chunk) | Chunk text, extraction model or prompt changes |
| 4. Embeddings | Cheap–moderate (local CPU) | Chunk text or embedding model changes |
| 5. Rule-key clustering | Seconds | **Always** |

| You change… | Single cache re-runs | Per-stage cache re-runs |
|---|---|---|
| Chunker size limit | Everything incl. OCR and extraction | Chunks → facts → embeddings for affected chunks; **OCR reused** |
| Extraction prompt | Everything | **Facts only** |
| `RULE_CLUSTER_THRESHOLD` | Everything | **Clustering only** |
| One document's content | That document | That document; unchanged chunks reused |

*Key trick — key each stage by its input, not the original file:* chunk key = `hash(stage-1 text) + chunker version`; fact key = `hash(chunk text) + extraction model + prompt version`; embedding key = `hash(chunk text) + embedding model`. Changes flow downstream automatically; if a stage re-produces identical output, everything after it is still reused.

*Consistency bonus:* LLM stages (OCR, extraction) stay fixed while later stages are changed, so the eval measures our change, not LLM noise (Q35).

*Trade-offs:* more moving parts (keep it simple: one table/folder per stage, key → JSON/binary). Stale-cache risk if a code change forgets a version bump → **prompt files hashed into keys automatically**; a `STAGE_VERSION` constant next to each stage's code; a test comparing a cold run with a cached run. Graders start cold, so this is for **our** development and eval cycle.

*Why atomic writes for the cache:* ingestion can be **interrupted mid-write** — container stopped (`docker compose down`, Ctrl+C, restart), process killed (out of memory, timeout), or two parallel extraction tasks writing the same key. The next start then reads a cut-off file:
- Truncated JSON → fails to parse (annoying but loud).
- Truncated binary (embeddings array, FAISS index) → may load **partly with no error** → missing chunks, wrong retrieval, **silently** — the dangerous case; or a crash on startup (must-pass #1).

Fix:
1. Write to a **temp file in the same directory** (`x.json.tmp`).
2. Flush + `fsync`.
3. `os.replace(tmp, final)` — atomic on the same disk (Linux and Windows). Readers see the old complete file or the new complete file, never a partial one.

Plus check on read: schema version per entry, checksum for binary files; unreadable / wrong version / bad checksum → **recompute and log**, never crash, never use it.

Simplest option: keep cache entries in a **separate SQLite cache DB** (not `veltrix.db`) — transactions are already atomic, so half-written entries can't exist; only the FAISS index file needs temp-file + rename.

*Why a separate SQLite cache DB (`cache.db`) — a design choice, not a PDF requirement:*

What the PDF says: §2.3 may add tables and indexes to `veltrix.db`, must not change existing ones; fresh DB copy at test time; §10.1 `./data` is replaced; §10.3 vector store local or in Docker Compose, only LLM/embedding APIs external; FAQ: holidays may be stored in a table if extracted automatically. **It never mentions a cache or a second DB** — neither required nor forbidden. A local file inside the container is allowed (same category as the FAISS index). The spec-minimal option (our tables in `veltrix.db`) would also pass; `cache.db` is a senior-level improvement.

| Reason | Detail |
|---|---|
| 1. Reviewers replace `veltrix.db` (§2.3, §10.1) | Their fresh copy has no cache tables — every swap would wipe a cache stored there |
| 2. Safe to delete | "Clear the cache" = `rm cache.db`, no risk to leave data; cold-path test is one command |
| 3. Locking | SQLite has one writer; thousands of ingestion writes would compete with `BEGIN IMMEDIATE` leave transactions (Q18) |
| 4. Clean boundary | `veltrix.db` holds their schema plus the minimum we add |

**What must stay in `veltrix.db`:** `proposals` (and `sessions`, `messages`). The double-"yes" guard (Q25) claims and deletes the proposal **in the same transaction** as the leave insert and balance update. Transactions are atomic only within one database file — with `ATTACH` in **WAL mode, transactions across attached databases are not atomic** together. → Anything written in the same transaction as leave data lives in `veltrix.db`.

| `veltrix.db` (source of truth, replaced by reviewers) | `cache.db` (disposable, outside `./data`) |
|---|---|
| `employees`, `leave_balances`, `leave_requests` (theirs, untouched) | `documents`, `chunks`, `policy_facts`, `holidays` |
| `sessions`, `messages`, `proposals` (ours, per §2.3) | Per-stage ingestion caches; FAISS index file alongside |

- Holidays are **read** from `cache.db` during `apply_leave` validation — fine, only the writes must be atomic together.
- Sessions reset when reviewers swap `veltrix.db` — correct: fresh DB, fresh conversations.

**Rule from the PDF: never commit `cache.db` or the FAISS index.** They contain corpus-derived facts — close to "seed data" (§4.1) — and would be stale on the modified corpus (must-pass #5). Add to `.gitignore`; build fresh at startup.

*DESIGN.md line (separate cache DB):* "Our derived data (documents, chunks, facts, holidays, stage caches) lives in a separate, disposable `cache.db` outside `./data` — a design choice; §2.3 would also allow these tables in `veltrix.db`. It survives reviewers swapping `veltrix.db`, can be cleared safely, and doesn't contend with leave writes for SQLite's single writer lock. Sessions, messages and proposals stay in `veltrix.db`, because the proposal is claimed in the same transaction as the leave write and SQLite transactions across attached databases are not atomic in WAL mode. Neither cache nor index is committed to the repository."

*DESIGN.md line (atomic writes):* "Cache writes are atomic — entries live in a separate SQLite cache DB (transactional), and the FAISS index is written to a temp file and swapped in with `os.replace` — so an interrupted ingestion never leaves a half-written entry. Entries carry a schema version and checksum; anything invalid is recomputed, never loaded."

*DESIGN.md line:* "Ingestion caches each stage separately (parse/OCR, chunks, facts, embeddings), each keyed by the hash of its input plus its own model/prompt/code version, so changes invalidate only downstream work — e.g. a new extraction prompt re-runs extraction but reuses OCR. Prompt files are hashed into keys automatically; rule-key clustering is always recomputed."

*Note on the summary line:* derived data lives in a separate `cache.db` (design choice); proposals, sessions and messages stay in `veltrix.db` for transactional atomicity with leave writes.

*Practical:*
- Stored **outside `./data`** (container path or named volume).
- **Atomic writes** (temp file → rename); unreadable or wrong-schema entries are **recomputed**, never a crash.
- **Graders start cold** — the cache doesn't help their first run; it helps restarts, eval runs and development. **Test the cold path**: clear the cache, confirm ingestion finishes well within 10 minutes (must-pass #1).

*DESIGN.md line:* "Ingestion caches expensive derived content (OCR text, chunks, facts, embeddings) per stage, keyed by sha256 of the file bytes plus pipeline, model and prompt versions — never by file name, because the modified corpus keeps names but changes content. Manifest metadata (access level, regions, dates, type) is never cached: it is read fresh at every startup, so an access change can't be masked by stale cache. Rule-key clustering is always recomputed; the cold path is tested against the 10-minute limit."

**Caches deliberately not used:**
- **Answer / semantic cache** — answers depend on role, region, reference date (§5) and live balances; a cached answer could leak data or go stale.
- **Tool-result cache** — balances change after apply/cancel; §8.2 requires re-validation at execution.

---

## Q8. The conversation is part of RAG — why can't we use memory? What does the PDF say?

**Answer:** We **do** use memory — conversation memory within a session is required. What we don't build is long-term memory across sessions. The PDF never forbids memory; it forces memory to be scoped to one session and one employee.

**The PDF requires in-session memory:**

| Section | What it says | What it means |
|---|---|---|
| §4.3 | "Earlier turns of the conversation" do **not** count toward the 3,000 tokens | History is expected to be sent to the LLM |
| §8.2 | "If the next message in the same session is a clear confirmation…"; "Make it until Friday instead" → new proposal | Must remember the pending proposal and its details |
| §8.3 | Clarify, then the user answers in the next turn | Must remember what was asked |
| §9.1 | "A new `session_id` starts a new conversation" | Memory is scoped to the session |
| §7.1 | "A session belongs to the employee who sent its first message" | Memory belongs to one employee only |

**The PDF limits memory to the session:**

| Section | Rule | Why long-term / shared memory breaks it |
|---|---|---|
| §9.1 | New `session_id` = new conversation | Carrying memory forward contradicts it |
| §10.3 | Tests run 3 times; output must be consistent | Memory from run 1 changes runs 2 and 3 |
| §7.1 | Ignore role/identity claims; identity only from DB | A stored claim could resurface; re-read role every turn |
| §7.4 | Send only personal data needed for the **current** request | Replaying old turns sends unneeded personal data |
| §7.2 / §7.3 | Never expose restricted docs or others' data | Shared memory is a leak channel |
| §5 | Answer follows docs applicable on the reference date | Remembered answers may be wrong for a new date |
| §8.2 | Proposal discarded on any other message; max one pending | Proposal memory must be short-lived |

**How memory works in the design:**
1. Per session: owner, last few turns (answer text only, no raw corpus chunks), at most one pending proposal.
2. Every turn: reload requester from DB, resolve reference date, re-run filtered retrieval, re-call tools for live data.
3. History helps the planner understand follow-ups ("that one", "make it Friday"); it never replaces retrieval or tools.

**README assumption:** "Conversation memory is scoped to a single session_id and its owning employee (§7.1, §9.1). We do not keep memory across sessions, to keep results reproducible across test runs (§10.3) and to avoid carrying personal data or unverified claims beyond the current conversation (§7.1, §7.4)."

---

## Q9. The employee is already identified — why not use long-term memory?

**Answer:** Knowing the employee removes the cross-employee leak (memory can be separated per employee), but the other reasons still apply.

1. **The header is not real authentication.** Login/authentication is out of scope (§3); `X-Employee-Id` is trusted, not verified. Within a session a mismatch → 403 (§7.1), but with long-term memory anyone sending `E1001` in a new session would get E1001's history.
2. **Tests run 3 times with the same employees (§10.3).** Their DB is reset between runs, but our memory store is not part of their DB — run 1's memory would change run 2's answers.
3. **§9.1 is explicit:** a new `session_id` starts a new conversation; long-term memory makes it behave like a continued one.
4. **Remembered facts go stale.** Balances change after apply/cancel; policies depend on the reference date (§5); the modified corpus changes values (must-pass #5) — stored answers would repeat old values.
5. **Claims and personal data persist.** "I'm in HR" must be ignored (§7.1); §7.4 limits personal data to the current request.
6. **It earns nothing.** No requirement asks for it; §1 values correctness and depth over breadth.

**Verdict:** In a real product with real login, preference-only memory (never facts or answers) with expiry and user deletion would be reasonable. In this assessment it is a net negative.

**DESIGN.md "considered and rejected":** "We considered per-employee long-term memory. Rejected: X-Employee-Id is trusted but not authenticated (auth is out of scope, §3), so long-term memory would expose history to anyone sending that ID; it would also break reproducibility across test runs (§10.3), contradict §9.1, and risk stale balances/policy values on a modified corpus. In production, with real authentication, we would add preference-only memory with expiry and user deletion."

---

## Q10. Why SQLite instead of Postgres?

**Answer:** The spec gives us SQLite and the test harness is built around that file. Postgres isn't forbidden, but it adds risk for no gain.

**Reasons from the PDF:**

| Section | What it says | Why it points to SQLite |
|---|---|---|
| §2 / §2.3 | We receive `veltrix.db`, a SQLite database | It is the source of truth — use it directly |
| §2.3 | "We will test with a fresh copy of the database, which may contain different rows" | They drop in a new `.db` file; the service must read it as-is |
| §10.1 | "We will replace `./data` with our own copy" (includes `veltrix.db`) | Postgres would need an import from their file on every startup |
| §10.2 | `DB_PATH`: path to `veltrix.db` | The config contract is a file path |
| §2.3 | Don't change or drop existing tables; may add tables and indexes | Easy in SQLite; Postgres means maintaining a copy of their schema |
| §12.1 #1 | `docker compose up` → `/health` 200 within 10 min, no manual step | One container is safer than two (no startup ordering, readiness waits or import step) |

**Hidden risk with Postgres:** if `veltrix.db` is imported into Postgres, every `apply_leave` / `cancel_leave` writes to Postgres, not their file. If their tests inspect `veltrix.db` after a write, they see no change — the service looks broken even when it works. With SQLite, writes land exactly where they expect.

**Does SQLite meet the technical needs?**

| Need | SQLite answer |
|---|---|
| Atomic apply/cancel (§6.2, §6.3) | Full ACID transactions; `BEGIN IMMEDIATE` so balance checks and updates can't race |
| Concurrent requests | WAL mode + busy timeout + one connection pattern — plenty for their test load |
| Extra tables (chunks, facts, holidays, sessions, proposals) | Allowed (§2.3); same file or a separate cache database |
| Speed | In-process, no network hop — faster than Postgres at this scale |

**Is Postgres allowed?** Technically yes — §10.3 restricts *external* services; a Postgres container inside Docker Compose isn't external, and pgvector is even listed as an option. Allowed but not justified: an extra moving part, an import step and the write-location risk.

**When we would switch:** multiple app instances writing at once, high-concurrency leave writes needing row-level locking, or 10,000 docs with pgvector replacing FAISS. That belongs in the 10,000-document section, not the build.

**DESIGN.md line:** "SQLite is used because the spec supplies `veltrix.db`, swaps in a fresh copy at test time (§2.3, §10.1) and configures it by file path (§10.2); writing directly to it keeps leave changes where the evaluators expect them. Apply/cancel use `BEGIN IMMEDIATE` transactions with WAL mode. At scale (multiple instances, concurrent writes, 10k docs) we would move to Postgres + pgvector."

---

## Q11. Why fastembed instead of OpenAI embeddings?

**Answer:** The spec allows either, but local fastembed removes a dependency that could fail must-pass #1. OpenAI embeddings aren't wrong — just riskier here.

**What the PDF says (§10.3):** "Any embedding model is fine. A local embedding model must run on CPU." / "The only external services allowed are LLM and embedding APIs." / "We will use our own API key." → a judgment call, not a spec rule.

**Why fastembed wins here:**

| Reason | Detail |
|---|---|
| 1. The key problem (main reason) | Anthropic has no embeddings API, so OpenAI embeddings would need a **second** key. Reviewers may supply only one → ingestion fails → `/health` never 200 → must-pass #1 fails. Local embeddings work with any single key. |
| 2. Ingestion reliability | No network calls, rate limits or outages during startup (10-min limit, §10) |
| 3. Determinism | Same model file, same vectors on all 3 test runs (§10.3); a hosted model can change |
| 4. Latency | Query embedding takes milliseconds on CPU, no round trip (30 s limit, §9.1) |
| 5. Data handling | HR-restricted docs never leave the container to be embedded |
| 6. Cost | Zero — matters at 10,000 docs and on re-indexing |
| 7. Small image | ONNX, no PyTorch; bge-small ≈ 130 MB, downloaded at build time |

**Honest trade-off:** OpenAI `text-embedding-3` is generally stronger than `bge-small`. It doesn't hurt here because there are 30 docs, retrieval is hybrid (BM25 catches exact terms), and pre-filtering shrinks the candidate set — chunking and filtering matter more than the embedding model. Prove it with recall@k in `make eval`; switch if recall is weak.

**Senior move — make it configurable:**
```
EMBEDDING_PROVIDER=local   # default: fastembed bge-small, works with any single LLM key
EMBEDDING_PROVIDER=openai  # optional: text-embedding-3-small, needs OPENAI_API_KEY
```
Listed in `.env.example` (§10.2).

**DESIGN.md line:** "Embeddings run locally (fastembed, bge-small, CPU) so ingestion depends only on the single LLM key reviewers supply — Anthropic has no embeddings API, and a missing second key would block /health. Local embeddings are also deterministic across test runs and add no network latency. Quality is backed by hybrid BM25 retrieval and measured with recall@k in the eval; `EMBEDDING_PROVIDER=openai` is available as an option."

---

## Q12. Why FAISS instead of Chroma?

**Answer:** Both are allowed and both would work. FAISS keeps the access filter in our own tested code and keeps one source of truth. Chroma means less code but more moving parts.

**What the PDF says (§10.3):** "Run the vector store locally or inside Docker Compose, for example FAISS, Chroma, Qdrant or pgvector." → a judgment call.

**Why FAISS wins here:**

| Reason | Detail |
|---|---|
| 1. One source of truth (main reason) | `access_level`, regions and `effective_date` already live in SQLite, where eligibility reads them. Chroma keeps its own copy — two copies of security-critical fields that can drift. FAISS stores only vectors and IDs. |
| 2. Security filter is our own tested code | `eligibility.py::allowed_chunk_ids` computes the allowed set; FAISS searches only those IDs. With Chroma the boundary becomes a `where` query in its filter syntax — harder to unit-test, easy to get subtly wrong (must-pass #2). |
| 3. Our filters don't map neatly to Chroma | `regions` is a list and dates need comparisons; Chroma's metadata filtering is built for simple scalar values → workarounds (per-region flags, integer dates). |
| 4. Exact search | `IndexFlatIP` is exact at 30 docs — no approximation, same results on all 3 runs (§10.3). Chroma defaults to approximate HNSW. |
| 5. Fewer dependencies | FAISS is one small library; `chromadb` pulls many packages and has had version conflicts and SQLite-version requirements in containers (risk to must-pass #1). |
| 6. No hidden network calls | Some Chroma versions send anonymous telemetry by default; §10.3 allows only LLM and embedding APIs as external services. FAISS makes none. |

**Honest trade-off:** Chroma gives built-in persistence and metadata filtering; FAISS costs ~50 lines (save/load the index, map FAISS IDs to chunk IDs). At this scale those lines buy a security boundary we fully control and can test.

**When to choose something else:** 10,000+ docs with rich filtering → Qdrant or pgvector (proper metadata indexes). Quick prototype → Chroma is fine.

**DESIGN.md line:** "FAISS (exact `IndexFlatIP`) stores only vectors and chunk IDs; all access metadata stays in SQLite, and search runs only over IDs returned by our tested `eligibility.allowed_chunk_ids`. This keeps one source of truth for security-critical fields and avoids depending on a vector DB's filter syntax for must-pass #2. Chroma was considered; at scale we'd move to Qdrant or pgvector with metadata indexes."

---

## Q13. Why BM25 together with FAISS?

**Answer:** Each catches what the other misses. HR questions mix exact terms with everyday paraphrases, so one retriever alone drops relevant chunks. The spec doesn't require hybrid search — it's a quality decision backed by numbers.

**What each is good and bad at:**

| | BM25 (keyword) | FAISS (embeddings) |
|---|---|---|
| Good at | Exact terms: "carry forward", holiday names ("Diwali", "Thanksgiving"), codes (`L5`), region codes, numbers, policy names | Meaning and paraphrase: "time off when I'm ill" → "sick leave"; "keep unused days?" → "carry-forward" |
| Bad at | Synonyms: "ill" never matches "sick" | Rare tokens, names, acronyms, numbers — a small model blurs them |
| Fails when | User's words differ from the document's | Query depends on one exact keyword |

**Why it matters in this corpus:**
- Holiday calendars (§2.1) are full of proper nouns and dates → BM25 strong.
- Policy questions are phrased casually → embeddings strong.
- Near-duplicates and versions (§2.1) share wording → both rank them close; precedence picks the winner (§5).
- Modified corpus (must-pass #5) may change wording → two retrievers make recall less fragile.

**How they're combined:**
1. **Both search only the allowed IDs** — BM25 must be restricted to `allowed_chunk_ids` just like FAISS; must-pass #2 applies to every retriever.
2. **Reciprocal Rank Fusion:** `score = Σ 1 / (60 + rank)` — merges by rank, so no calibrating BM25 scores against cosine; simple, no tuning, deterministic.
3. Collapse near-duplicate clusters → precedence resolver → 3,000-token packer.

**Honest trade-off:**
- Cost: ~30 lines + `rank_bm25`; microseconds at 30 docs.
- Subtle point: BM25's IDF is computed over the whole corpus, restricted docs included. This affects ranking only — no restricted text is retrieved or sent. Mention it, or compute IDF per access level to be strict.
- No cross-encoder reranker by default (latency, heavier CPU model); add only if the eval shows ranking problems.

**Senior move — ablation in the eval, results in DESIGN.md:**

| Retriever | recall@5 | Citation precision |
|---|---|---|
| BM25 only | … | … |
| FAISS only | … | … |
| Hybrid (RRF) | … | … |

If hybrid wins, the choice is justified with data; if not, it's an honest finding (§1).

**DESIGN.md line:** "Hybrid retrieval: BM25 catches exact terms (holiday names, policy terms, codes) that a small embedding model blurs; FAISS catches paraphrases that keywords miss. Both search only the eligible chunk IDs, and results are merged with Reciprocal Rank Fusion (rank-based, no score calibration, deterministic). An ablation in the eval (BM25 vs dense vs hybrid recall@5) backs the choice."

---

## Q14. Why RRF instead of weighted score fusion?

**Answer:** RRF needs no tuning and doesn't break when the corpus changes. Weighted score fusion needs labelled data to tune, and that tuning drifts on the modified corpus.

**How the two work:**
- **Weighted fusion:** `score = α · norm(BM25) + (1 − α) · cosine` — normalize two score scales and pick α.
- **RRF:** `score = Σ 1 / (k + rank)`, k = 60 (standard value from Cormack et al., 2009) — positions only, never raw scores.

**Why RRF fits better here:**

| Problem with weighted fusion | Why it hurts this project |
|---|---|
| 1. Incompatible score scales | BM25 is unbounded and depends on query length and corpus statistics; bge-small cosine clusters in a narrow band (~0.6–0.85). Adding them makes no sense without normalization. |
| 2. Unstable normalization | Per-query min-max is thrown off by one outlier; pre-filtering leaves small candidate sets, which makes it worse. |
| 3. α needs labelled data | Our eval set is small (~30–50 questions) → a tuned α overfits to our own test cases. |
| 4. Drift on the modified corpus (must-pass #5) | Changed docs change BM25's IDF and shift score distributions; an α tuned on our corpus may be wrong on theirs. Ranks stay meaningful. |
| 5. Determinism | Fixed k + ties broken by `chunk_id` → identical results across the 3 runs (§10.3). |
| 6. Simplicity | One line, no parameter to defend — avoids "how did you choose α?" |

**Honest trade-off:**
- RRF loses score magnitude: a very confident BM25 hit and an unsure one both count as rank 1.
- With a large labelled set and a stable corpus, tuned weighted fusion can beat RRF — not our situation.
- It rarely matters here: precedence and the token packer decide what reaches the LLM, so small top-10 ranking differences rarely change the answer.

**Senior move — add a row to the ablation:**

| Retriever | recall@5 |
|---|---|
| BM25 only | … |
| FAISS only | … |
| Weighted (α = 0.5, min-max) | … |
| RRF (k = 60) | … |

Keep RRF unless weighted wins clearly; even then, note the overfitting risk on a small eval set.

**DESIGN.md line:** "Fusion uses RRF (k=60) rather than weighted score fusion: BM25 and cosine scores live on incompatible scales, normalization is unstable on small pre-filtered candidate sets, and tuning α on a small eval set would overfit and drift on the modified corpus. RRF is parameter-light and deterministic; a weighted variant is included in the ablation for comparison."

---

## Q15. Why tiktoken for token counting?

**Answer:** The spec requires it — not our choice. The senior part is knowing what to count, where to count it, and the traps.

**What the PDF says (§4.3):** "Each LLM call may include at most 3,000 tokens of corpus text. Count tokens with the `tiktoken` encoding `cl100k_base`, whichever LLM you use." → it is a measuring standard; evaluators recount with the same encoding (must-pass #4).

**What it is and isn't:**
- **Is:** the official ruler for the 3,000-token corpus cap.
- **Isn't:** Claude's real tokenizer — the count won't match billing or context-window usage.
- **So:** use `cl100k_base` for the cap and its logs; log the provider's reported token usage separately for cost/latency.

**What counts (§4.3):**

| Counts toward 3,000 | Doesn't count |
|---|---|
| Retrieved chunks (incl. OCR text, tables as markdown) | System prompt instructions |
| Summaries or notes made from the corpus | Earlier conversation turns |
| Document text stored in the database | Structured tool records (e.g. extracted holiday dates) |

**Grey area:** chunk labels such as `[chunk 12 · POL-LEAVE-V2 · §3.1]` — count them as corpus text too. Conservative and cheap.

**How it's enforced:**
1. `context_packer.py` counts each chunk and stops before exceeding 3,000; never cut mid-sentence — skip the chunk.
2. `LLMClient` recounts every corpus segment just before sending and refuses any call over 3,000 — the single chokepoint.
3. Corpus text travels as a typed `CorpusSegment`, separate from instructions.
4. `corpus_tokens` logged on every LLM call (§4.3), including the planner (0).
5. Property test: random chunk combinations never exceed 3,000.

**Traps to catch:**
- tiktoken **downloads the encoding file on first use** → pre-download at Docker build time and set `TIKTOKEN_CACHE_DIR` (protects must-pass #1).
- Load the encoding once and reuse it.
- **Ingestion LLM calls count too** — fact extraction gets one chunk at a time, well under 3,000.
- The OCR vision call sends an image, but its **output** is corpus text — any later call including it counts it.

**DESIGN.md line:** "Corpus tokens are counted with tiktoken `cl100k_base` as §4.3 requires (a measurement standard, not Claude's tokenizer). Corpus text travels as typed segments; the context packer fills to the budget and `LLMClient` recounts and refuses any call above 3,000, logging `corpus_tokens` per call (ingestion calls included). The encoding is pre-cached at Docker build time to avoid a network download at startup."

---

## Q16. Why FastAPI instead of Flask?

**Answer:** The spec allows any framework. FastAPI gives the exact API contract, request validation and async I/O with the least code. Flask would work too — but FastAPI has traps that must be handled; catching them is the senior part.

**What the PDF says:** §1 "Frameworks and libraries: any you like." §9 "match this contract exactly, because our tests are automated."

**Why FastAPI fits:**

| Need (spec) | FastAPI | Flask |
|---|---|---|
| Exact request/response shape (§9.1) | Pydantic models + `response_model` validate every response | Manual dicts or extra libraries (marshmallow) |
| 400 for missing/invalid body (§9.1) | Automatic validation (see trap 1) | Hand-written checks |
| Slow LLM I/O within 30 s (§9.1) | Async — concurrent requests don't block while waiting on the LLM | Synchronous by default (WSGI); needs extra workers/threads |
| Background ingestion + `/health` 503 until ready (§9.2) | `lifespan` hook + ready flag | Possible, more manual |
| Integration tests for `/chat` (§11.1) | `TestClient` / `httpx` built in | `app.test_client()`, also fine |
| Reviewer experience | Free OpenAPI docs at `/docs` | None by default |

**Traps:**
1. **422 vs 400** — FastAPI returns HTTP 422 for validation errors; the spec says 400. Override the `RequestValidationError` handler.
2. **Missing header → 401, not 422** — a required header parameter returns 422 when missing. Read `X-Employee-Id` as optional and raise 401 yourself for missing and unknown IDs.
3. **Blocking code in `async def`** — SQLite, FAISS and CPU embedding block the event loop. Use plain `def` endpoints (thread pool) or `run_in_threadpool`; keep `async` for LLM HTTP calls.
4. **Order of checks** — 400 body → 401 employee → 403 session, as §9.1 lists them; test each.

**Honest trade-off:** Flask is simpler, familiar and would pass the spec. FastAPI's defaults (422, async) can hurt if unknown; its value comes only when the traps are handled.

**DESIGN.md line:** "FastAPI with Pydantic response models enforces the §9 contract on every response. Default 422 validation errors are remapped to 400, and a missing/unknown X-Employee-Id returns 401 via explicit handling. Blocking work (SQLite, FAISS, embeddings) runs in the thread pool so the event loop stays free for LLM I/O; ingestion runs in a lifespan background task behind the /health ready flag."

---

## Q17. Why Pydantic instead of dataclasses?

**Answer:** Not either/or — **Pydantic at the boundaries** (where untrusted data enters), **frozen dataclasses inside** (where data is already trusted). A dataclass has type hints but no runtime validation, and this project has several untrusted inputs.

**Key difference:**

| | Pydantic | Dataclasses |
|---|---|---|
| Runtime type validation | Yes | No — hints only |
| Parses JSON into objects | Built in | Manual |
| Clear error messages | Yes | No |
| Generates JSON Schema | `model_json_schema()` | No |
| Speed / dependencies | Slower, external library | Fastest, standard library |

**Where Pydantic is required (boundaries):**

| Boundary | Why validation matters |
|---|---|
| 1. HTTP API (§9) | FastAPI-native; bad body → 400 (remapped from 422); response models enforce the contract; `route` is a `Literal[...]` of the six routes |
| 2. LLM output (biggest reason) | Planner, composer and fact-extraction JSON can be malformed or wrong → validate → one repair retry → safe fallback |
| 3. LLM-chosen tool args (§6.1) | `leave_type: Literal["annual", "sick"]`, real dates — invalid args fail before tool logic runs |
| 4. Tool schemas for the LLM | `model_json_schema()` generates tool definitions from the same models — schema and validation never drift; exact names (§6.1) |
| 5. Config (§10.2) | `pydantic-settings` validates env vars (e.g. `APP_TODAY` as a date), clear startup error |
| 6. manifest.json (§2.2) | Modified manifest may have unexpected values — validate `doc_type`, `access_level`, date format; log bad entries |

**Where dataclasses are better (internals):** `eligibility.py`, `precedence.py`, `working_days.py` receive already-validated data → `@dataclass(frozen=True)`: faster in hot loops, immutable (no accidental edits to precedence inputs), no dependency, easy to unit-test.

**Traps:**
- **Lax coercion** — `"12"` → `12`, `"yes"` → `True` by default. Use **strict mode** for LLM output and tool args.
- **Personal data in serialization** — use a field serializer or a separate model for `tool_calls` args so emails are always masked and phones never appear (§7.4).
- **Hidden 422s** — see Q16.

**DESIGN.md line:** "Pydantic validates everything that crosses a trust boundary — HTTP requests/responses, LLM JSON output, LLM-chosen tool arguments (strict mode), config and the manifest — and generates the tool schemas sent to the LLM from the same models. Inside, pure domain logic (eligibility, precedence, working days) uses frozen dataclasses: already-validated, immutable and dependency-free."

---

## Q18. Why SQLite transactions with `BEGIN IMMEDIATE`?

**Answer:** The spec requires one transaction (§6.2, §6.3); `BEGIN IMMEDIATE` makes that transaction actually safe. A plain `BEGIN` still allows a race where the balance goes negative.

**What the PDF says:**
- §6.2: create the request, store `working_days`, subtract the balance "in one database transaction."
- §6.3: set status to cancelled and add the days back "in one database transaction."
- §8.2: execution status is `success` or `error` "because the data may have changed since the proposal" → re-check rules **inside** the transaction.

**The problem — check, then act:** apply reads balance and requests → checks → inserts and subtracts. Two simultaneous requests for one employee (double-click, two sessions):

| Time | Request A | Request B |
|---|---|---|
| t1 | reads balance = 5 | reads balance = 5 |
| t2 | needs 3 → OK | needs 3 → OK |
| t3 | inserts → balance 2 | inserts → balance −1 ❌ |

Result: negative balance, possibly overlapping leave, both "succeeded". Same with cancel: two cancels refund twice.

**Why plain `BEGIN` doesn't fix it:** SQLite's default `BEGIN DEFERRED` takes no lock until the first write → both read and pass checks → one hits `SQLITE_BUSY`; in WAL the second can fail with a stale-snapshot error that `busy_timeout` doesn't retry. Either the race or random failures under load.

**What `BEGIN IMMEDIATE` does:**
- Takes the write lock **at the start**, before the reads.
- Only one apply/cancel runs its read-check-write at a time; the second waits (`busy_timeout`), reads the **updated** balance, and correctly rejects.
- With WAL, readers aren't blocked — `/chat` reads and retrieval keep working.

```python
conn.execute("BEGIN IMMEDIATE")
try:
    # re-read balance and requests, re-validate every rule
    # insert request (uuid4, 'pending'), subtract working_days
    conn.execute("COMMIT")
except Exception:
    conn.execute("ROLLBACK")
    raise
```

**Python traps:**
1. Python's `sqlite3` module silently issues its own `BEGIN` with the default `isolation_level` → on Python 3.11 open with `isolation_level=None` and control transactions explicitly.
2. Set `PRAGMA journal_mode=WAL`, `PRAGMA busy_timeout=5000`, `PRAGMA foreign_keys=ON`.
3. One connection per thread — FastAPI runs `def` endpoints in a thread pool.
4. All validation **inside** the transaction; checking before `BEGIN` brings the race back.

**Alternative considered:** conditional `UPDATE ... SET balance_days = balance_days - ? WHERE ... AND balance_days >= ?` is atomic for the balance but doesn't cover the overlap check or double cancel. `BEGIN IMMEDIATE` protects every rule with one mechanism.

**Overkill for their test load?** Tests may be sequential, but it costs one keyword, reviewers look for unlocked check-then-act bugs, and a test proves it: two threads apply at once with balance for only one → exactly one succeeds, balance never negative.

**DESIGN.md line:** "Apply and cancel run as `BEGIN IMMEDIATE` transactions: the write lock is taken before the balance and overlap checks, so concurrent requests can't both pass validation (no negative balances, overlaps or double refunds). All rules are re-validated inside the transaction, as §8.2 expects data may change after a proposal. Connections use WAL, a busy timeout and `isolation_level=None` so Python's sqlite3 module doesn't open its own transactions. A two-thread concurrency test covers it."

---

## Q19. Why RRF with k = 60?

**Answer:** k = 60 is the standard default from the original RRF paper and is known to be robust without tuning. We have no labelled data to tune it, so a well-tested default is the defensible choice — with one trade-off to understand and check.

**What k does:** `score = Σ 1 / (k + rank)` — k controls how much a top rank counts versus agreement between the two lists.

| k | rank 1 | rank 2 | rank 10 | Behaviour |
|---|---|---|---|---|
| 1 | 0.500 | 0.333 | 0.091 | Top-heavy: whichever retriever ranks a chunk first mostly decides |
| 60 | 0.0164 | 0.0161 | 0.0143 | Flat: being found by **both** retrievers matters most |

With k = 60, a chunk ranked 5th in both lists scores `2/65 = 0.031`, beating a chunk ranked 1st in only one list (`1/61 = 0.016`) → **agreement wins**.

**Why k = 60:**
1. **Source:** Cormack, Clarke & Büttcher (SIGIR 2009) — worked well across many TREC collections, not very sensitive to the exact value; the standard default since.
2. **No labelled data** — same argument as α (Q14): tuning on 30–50 questions overfits and may drift on the modified corpus (must-pass #5).
3. **Fits hybrid search** — agreement between BM25 and embeddings is a strong relevance signal.
4. **Determinism** — fixed constant + tie-break by `chunk_id` → same order every run (§10.3).

**Honest trade-off:** a chunk found by only **one** retriever can be pushed down (e.g. an exact holiday name: BM25 ranks it 1st, embeddings miss it; several so-so chunks found by both may outrank it). Acceptable because:
- Top 8–10 fused results go to precedence and packing, so a BM25 rank-1 hit still makes the set.
- Precedence and the 3,000-token packer decide what reaches the LLM.
- Holiday dates for working days come from the `holidays` table, not retrieval.

**Senior move — sensitivity check in the eval:**

| k | recall@5 |
|---|---|
| 10 | … |
| 30 | … |
| 60 | … |

Similar results → confirms the paper; keep 60 and say so. Smaller k clearly better → BM25-only exact hits matter here; report and decide.

**DESIGN.md line:** "RRF uses k=60, the standard value from Cormack et al. (2009), chosen because it's robust without tuning and we have no labelled data to tune it without overfitting. k=60 favours chunks both retrievers agree on; single-retriever exact hits are protected by passing the top 8–10 fused results forward, and holiday lookups bypass retrieval entirely. A k ∈ {10, 30, 60} sensitivity check is in the eval."

---

## Q20. Why section-based chunking?

**Answer:** The spec's citation rule is section-based, so chunking by section makes every citation exact. It also keeps each rule with its conditions, which precedence depends on.

**What the PDF says (§4.2):** each citation needs a `locator` — "the nearest section heading, or the page number (for example `p. 3`) when the document has no headings." One chunk = one section → its locator is that heading.

**Why it fits:**

| Reason | Detail |
|---|---|
| 1. Exact citations (§4.2) | One chunk → one section → one locator. A fixed-size chunk can span two headings — which one to cite? |
| 2. Rule stays with its conditions | "12 sick days per year. A medical certificate is required after 2 consecutive days." Fixed-size splitting can cut it in half. |
| 3. Precedence needs whole rules (§5) | Fact extraction per chunk → complete facts (value, unit, conditions) → correct conflict detection |
| 4. Fits the token budget (§4.3) | ~350 tokens (max 500) → winner, conflicting version and upcoming change all fit in 3,000 |
| 5. Tables stay whole | Kept within their section, never split mid-row |
| 6. Robust on the modified corpus (must-pass #5) | Headings are structure; changed values don't move boundaries (fixed offsets do) |
| 7. Deterministic | Same document → same chunks every run |

**Alternatives considered:**

| Approach | Why not |
|---|---|
| Fixed-size (e.g. 500 tokens + overlap) | Splits rules, unclear locators, overlap wastes budget |
| Semantic chunking (embedding-based boundaries) | Boundaries ignore headings → unclear locators; slower; extra tuning knob |
| Whole document | ~1 page each, but 3,000 tokens holds only 4–5 docs — conflicting versions may not fit; vague citations |
| Proposition / LLM chunking | LLM call per passage — slow, costly, risks the 10-min ingestion |

**Edge cases to handle:**
- **No headings** → page/paragraph chunks with `p. N` locators (§4.2).
- **Long sections (> 500 tokens)** → split by paragraph with small overlap; every piece keeps the same locator.
- **Very small sections** → keep as their own chunks; don't merge across headings.
- **Heading detection per format:** Markdown `#`, HTML `h1`–`h6`, DOCX heading styles are easy; **PDF is hardest** — font size/bold heuristics, fall back to page-based chunks.
- **OCR'd scan** → vision model asked to keep headings; else fall back to `p. N`.
- **Contextual header** → prefix the heading path (`Leave Policy v2 › Sick Leave › Certification`) to improve BM25 and embeddings. These words count toward the 3,000 tokens — keep them short.

**DESIGN.md line:** "Chunks follow section boundaries (~350 tokens, max 500) so each chunk's citation locator is its nearest heading (§4.2), each rule stays with its conditions for fact extraction and precedence, and the winner, conflicting and upcoming versions fit together in the 3,000-token budget. Documents without headings fall back to page-based chunks with `p. N` locators; long sections are split by paragraph while keeping their locator. Fixed-size, semantic and whole-document chunking were considered and rejected."

---

## Q21. Why a vision LLM for OCR?

**Answer:** The spec allows either (§4.1), and a vision LLM reads tables and headings in a scan far better than Tesseract. But a vision LLM can invent text, so it needs guards — the guards are the senior part.

**What the PDF says:** §2.1 "At least one scanned document with no text layer. You will need OCR to read it." §4.1 "You may use any OCR library or a vision-capable model." §4.1/§13: ingestion must be fully automatic — no typing out the scan.

**Why vision LLM is primary:**

| Reason | Detail |
|---|---|
| 1. Tables | Returns a markdown table with rows/columns intact; Tesseract jumbles columns |
| 2. Headings → citations (§4.2) | Marks headings so section chunking and locators work like any other document |
| 3. Messy scans | Handles skew, noise, stamps, low contrast without hand-tuned preprocessing |
| 4. Pipeline-ready output | Markdown goes straight into the block model and chunker — no OCR-specific code downstream |
| 5. No extra key | Claude and OpenAI both accept images — same single key reviewers supply |

**Risks and guards:**

| Risk | Guard |
|---|---|
| **Hallucination** — may "correct" or invent text, especially numbers (critical: facts drive answers) | Verbatim prompt ("do not correct, do not summarize, mark unreadable as `[illegible]`"); **cross-check numbers and dates against Tesseract**; mismatches flagged low-confidence in the ingestion report and DESIGN.md |
| **Prompt injection inside the image** (§2.1, §7.5) | OCR call has no tools and a transcription-only instruction; output treated as corpus data and scanned by `injection_scan.py` |
| **Run-to-run variation** (§10.3) | Temperature 0 + ingestion cache by file hash — read once, identical text on all 3 runs |
| **API failure / timeout** (10-min limit) | Tesseract fallback — `/health` still reaches 200 |
| **Token budget** | Call sends an image, not corpus text; its output is corpus text and counts in later calls (Q15) |

**Why not Tesseract primary:** local, deterministic, free, fast — but weak on tables and layout, no heading structure, needs preprocessing (deskew, binarize) on poor scans. Kept as **fallback and number cross-check**.

**The Tesseract number cross-check in detail:**

*Why numbers and dates, not all text:*
1. **Where errors do damage** — "12 sick days" read as "10" is a wrong policy answer; a wrong holiday date gives a wrong working-day count. A misspelled word does little harm.
2. **Where vision LLMs tend to err** — tempted to "fix" or fill in plausible values.
3. **Mechanically comparable** — whole-text comparison between engines is noisy; the set of numbers and dates on a page either matches or doesn't.

*Why Tesseract, not a second vision call:* **independent errors.** The same model run twice repeats the same biases. Tesseract uses different technology, so its mistakes are unrelated to the LLM's — when both agree, the value is very likely right. Its own weaknesses (0/O, 1/l, 5/S, jumbled tables) matter less because it only checks values, not structure.

*How:*
1. Run both engines on the same rendered page (~300 DPI).
2. Extract and normalize numbers and dates (digits only, ISO dates, "twelve" → 12, no thousands separators).
3. Compare the two multisets per page.
4. Agree → high confidence.
5. Disagree → neither wins automatically (Tesseract can be wrong too); keep the vision output (better structure); mark the page and its facts **low-confidence**; list mismatches ("vision 12, Tesseract 10, p. 1"); optionally re-run vision once at higher resolution.
6. Every mismatch goes into DESIGN.md §6 Failure modes — an honest finding §1 rewards.

*Cost and limits:* ~1 s per scanned page, local. Both engines agreeing on the same wrong value is possible but unlikely (independent errors). Pages without numbers have nothing to check. It **detects** errors, it doesn't **fix** them — state that plainly.

**Practical details:**
- Detect scanned pages **per page** (text below a character threshold) — handles mixed documents.
- Render at ~200–300 DPI with `pypdfium2`; lower resolution hurts small digits.
- Ingestion report records OCR engine per page, cross-check result and warnings.

**DESIGN.md line:** "Scanned pages (detected per page by an empty text layer) are rendered at ~300 DPI and transcribed by a vision LLM with a verbatim-only, tool-less prompt, producing markdown with headings and tables so chunking and citations work unchanged. Numbers and dates are cross-checked against Tesseract — an independent engine whose errors are uncorrelated with the LLM's — and mismatches mark the page and its facts low-confidence in the ingestion report and failure modes (detection, not correction). OCR output is cached by file hash for reproducibility and treated as untrusted corpus text. Tesseract is the fallback if the vision call fails."

---

## Q22. Why rule extraction at ingestion?

**Answer:** §5 precedence only works if the system knows two documents state "the same rule" — and the spec says recognising that is part of the task. Doing it once at ingestion turns it into data the code can compare, test and resolve deterministically.

**What the PDF says:**
- §5.1: "When eligible documents give different values for the same rule, apply these tests in order…"
- FAQ §13: "What counts as 'the same rule'? … such as the number of sick days. Recognising this is part of the task."
- §5.1 upcoming changes: mention a future document only if it "would win under the rules above on its effective date."
- §4.1: no facts written by hand → recognition must be automatic.

**Why at ingestion, not at question time:**

| Problem at question time | How extraction solves it |
|---|---|
| Token budget (§4.3): comparing needs every version (v1, v2, FAQ, addendum, upcoming v3) in context | Facts compared in code before the LLM call; only winning and conflicting passages sent |
| Non-determinism (§10.3): LLM may pick different winners across 3 runs | `precedence.resolve` sorts typed facts — same winner every time |
| Not testable | The spec's worked example becomes a unit test |
| Upcoming changes need a "would it win?" check | Re-run `resolve` at the future doc's date |
| Retrieval may miss a version (top-k) | Facts looked up by `rule_key` across all eligible docs |

**How it works:**
1. **Extract per chunk:** JSON schema `{topic, attribute, value, unit, conditions, quote_span}` (e.g. `sick leave · annual allowance · 12 · days`).
2. **Verify:** `quote_span` must exist verbatim in the chunk; otherwise drop and log.
3. **Group:** embed fact keys, cluster (same unit required) into one `rule_key` — learned from the corpus, no hand-written taxonomy.
4. **Resolve** per rule at question time with the requester's eligible docs and reference date.
5. **Facts select, chunks answer:** the LLM answers from source text with citations; facts decide which passages it sees and flag conflicts/upcoming changes.

**Risks and guards:**

| Risk | Guard |
|---|---|
| Extraction errors (wrong value/unit) | Quote verification, temperature 0, cached by content hash |
| Over-merging ("annual leave days" vs "carry-forward days") | Same unit required, attribute compared separately, conservative threshold — better to miss a merge than make a wrong one |
| Under-merging (conflict missed) | Composer still sees chunks and flags visible disagreements; eval cases built around known conflicts |
| Conditions (L1–L3 vs L4+, tenure) | `conditions` are part of the fact — different conditions = different rules, not a conflict |
| Ingestion time/cost | ~30 docs, ~150 chunks; parallel with limited concurrency, cached |

**Alternatives considered:** LLM compares versions at question time (budget, non-deterministic, untestable); regex/keyword rules (brittle on modified corpus, close to hardcoding); no conflict detection (fails §5).

**Quote verification in detail:**

*Why it matters:* fact extraction is the one place where LLM output becomes **trusted data** — precedence, conflicts and upcoming changes are decided in code from facts (Q23). An invented fact silently poisons every answer about that rule, consistently across all 3 runs. Quote verification is the check between "the LLM said so" and "the document says so".

*What goes wrong without it:*

| LLM error | Example |
|---|---|
| Invented value | Document says 12 days; fact says 10 |
| Derived value | "1.5 days per month" → "18 days per year" — not what the document states; caps or pro-rating may change it |
| Merged sentences | Value from one rule, condition from another |
| Wrong document | Value carried over from a similar chunk in the same batch |
| Paraphrased condition | "after 2 days" → "after 2 working days" |

*The two checks:*
1. **Quote exists** — `quote_span` appears in the chunk after light normalization (whitespace, quote characters, dashes, case, table pipes). **Exact** after normalization — no fuzzy matching, which would let paraphrases through.
2. **Value is in the quote** — the fact's `value` (and unit, if present) appears in that quote; blocks real quotes paired with values taken elsewhere or computed.

*Rule:* values extracted **exactly as stated** ("1.5 days per month" stays 1.5, unit days/month); arithmetic happens later in code (Q38).

*When a check fails:* drop the fact and log the reason (no repair); **the chunk is kept** — still retrievable and answerable as text, only the precedence aid is lost. Ingestion report counts extracted / verified / dropped by reason; a high drop rate for one document (OCR problem, odd table) goes into failure modes.

*Free extras:* reliable citation locators (quote pins the text); deterministic (pure code); reviewer trust ("every stored fact is backed by a verbatim quote").

**Embedding clustering for rule keys in detail:**

*The problem:* the LLM names the same rule differently — "sick leave · annual allowance" (policy v2), "sick days · days per year" (FAQ), "medical leave · entitlement" (DE addendum). Unless they become one `rule_key`, precedence never compares them and the conflict is invisible.

*Options considered:*

| Option | Problem |
|---|---|
| Exact string match on LLM keys | Fails immediately — naming is inconsistent |
| Fixed hand-written topic list | Close to hardcoding (§4.1); breaks when the modified corpus adds a rule |
| One big LLM call grouping all facts | Facts are "notes made from the corpus" → count toward 3,000 tokens (§4.3); may not fit; not reproducible |
| LLM judging every pair | n² calls — slow, costly, doesn't scale |
| **Embedding clustering (chosen)** | Local, cheap, deterministic, incremental, nothing hand-written |

*What goes into the key:*

| Field | In the key? | Why |
|---|---|---|
| topic + attribute | ✅ embedded | The "what" of the rule |
| unit | ✅ must match | "days per year" ≠ "days per month" |
| conditions | ✅ must match | L1–L3 vs L4+ are different rules, not a conflict |
| **value** | ❌ **never** | Same rule with different values must land together, or conflicts can never be found |
| region, effective date, doc type | ❌ | Handled by precedence (scope, recency, authority) |

*How:* partition facts by `(unit, conditions)` → within each partition, embed `"topic: attribute"` with local bge-small → **average or complete linkage** at a **conservative** threshold (not single linkage, which chains unrelated rules into one giant cluster) → fixed input order for reproducibility. Optional: pairs just below the threshold get a small, bounded, cached LLM yes/no check ("same rule?"), counted in the token budget.

*Why average linkage, not single or complete:*

| Linkage | Merges when… | Risk |
|---|---|---|
| Single | **Any one** pair across groups is similar | **Chaining** → over-merging |
| Complete | **Every** pair is similar | Strictest; can split real synonyms over one unusual phrasing |
| **Average (chosen)** | **Mean** similarity across all pairs is high | Balanced — tolerates an outlier, rejects chains |

Chaining example — each neighbour is similar: "annual leave allowance" ~ "annual leave carry-forward" ~ "carry-forward cap" ~ "sick leave carry-forward". Single linkage merges all four into one rule → precedence compares a 20-day allowance with a 5-day sick carry-forward cap and picks a "winner" — silent over-merge. Average linkage sees the far-apart pairs, the mean drops, the chain breaks. Complete linkage also prevents chaining but can split "medical leave entitlement" from "sick days per year" over one far pair → real conflict missed. **Plan:** average by default; switch to complete (one setting) if the ingestion report or eval shows over-merges.

*Why complete is the fallback:* over-merging is the worse failure (silent wrong winner); under-merging is safer (composer can still flag the disagreement). Complete linkage merges only if **every** pair is similar, so it can't over-merge through one weak link — its errors fall on the safe side. Not the default because it's too strict for synonyms: one far pair ("medical leave entitlement" vs "sick days per year") blocks the merge and a real conflict goes undetected.

*Switch linkage or tighten the threshold?*

| Fix | Effect |
|---|---|
| Tighter threshold (average) | Blocks **all** merges a bit more, good ones included — blunt |
| Complete linkage | Blocks merges only where some member is far from the rest — the pattern of a chained over-merge — targeted |

Order: read the ingestion report to see **why** it over-merged → a chain → complete linkage; a generally loose threshold → tighten it; close calls → LLM yes/no check for near-threshold pairs. Switch only on **evidence** (a cluster mixing different rules in the report, or a wrong winner in an eval case), never in advance. Implementation: `scipy.cluster.hierarchy` — `linkage(X, method="average", metric="cosine")` + `fcluster(Z, t, criterion="distance")` — trivial at ~150–300 facts, deterministic with fixed input order; run **within each (unit, conditions) partition** (see below).

*Why agglomerative clustering, and why scipy:*

| Need | Agglomerative | K-means | DBSCAN | HDBSCAN |
|---|---|---|---|---|
| Number of rules unknown | ✅ distance threshold, no k | ❌ must choose k | ✅ | ✅ |
| Cosine + average linkage | ✅ | ❌ Euclidean, centroid-based | ✅ | ✅ |
| No chaining | ✅ (average/complete) | n/a | ❌ behaves like single linkage | ⚠️ partly |
| Groups of 2 common (v1 vs v2) | ✅ | ⚠️ | ⚠️ needs min size | ❌ min cluster size drops pairs as noise |
| Deterministic | ✅ no random start | ❌ random init | ✅ | ✅ |
| Interpretable threshold | ✅ + dendrogram for choosing it | ❌ | ⚠️ `eps` | ❌ |

Key rows: unknown number of rules, many groups of just 2 facts, chaining must be avoided → only agglomerative with average linkage meets all three.

**scipy, not sklearn:** sklearn isn't otherwise in the stack and pulls in scipy anyway; `scipy.cluster.hierarchy` does exactly the same (`linkage` + `fcluster`) and draws the dendrogram. Minimal-stack choice (TECH_STACK principle). Don't hand-write it — tested library code.

**Hard constraints before clustering, not after:** first **partition** facts by `(unit, conditions)`, then cluster **within each partition**. Incompatible facts can never share a group; splitting afterwards could leave odd fragments, and averaging distances across groups that should never merge distorts the result.

*DESIGN.md line:* "Rule keys are clustered hierarchically (scipy, average linkage, cosine distance, distance threshold) because the number of rules is unknown, many rule groups are just two versions, and chaining must be avoided — ruling out k-means, DBSCAN and HDBSCAN. Facts are first partitioned by unit and conditions, so incompatible facts can never merge."

*The threshold value — **to be measured**, not chosen in advance:*

Why no number yet: (1) similarity scales are model-specific — bge-small gives compressed, high cosine scores (even fairly unrelated short phrases often ~0.6–0.8); (2) fact keys are very short, pushing scores closer together; (3) it depends on how the LLM names things in **this** corpus. A number picked now has no evidence behind it ("why 0.85?").

How to choose it (defensible method):
1. Run extraction on the real corpus; compute all pairwise similarities between fact keys (trivial at 150–300 facts).
2. Hand-label a small reference set: ~15–20 clearly **same**-rule pairs and ~15–20 clearly **different** but similar-sounding pairs ("annual leave allowance" vs "annual leave carry-forward"). Labels only calibrate the threshold — no facts hardcoded (§4.1).
3. Plot both groups' scores and look for the **gap**.
4. Set the threshold inside the gap, **leaning strict** (over-merging is worse).
5. Stability check: move it ±0.03; if clusters change a lot, it's fragile on the modified corpus → add the LLM yes/no check for near-threshold pairs.
6. Validate with eval known-conflict cases and by reading the clusters in the ingestion report.

Practical notes:
- The scipy `fcluster` threshold `t` (with `criterion="distance"`) is a **distance**: with `metric="cosine"` it is `1 − similarity` (similarity 0.88 → `t=0.12`).
- With average linkage the threshold applies to the **mean** distance between groups, not one pair.
- Configurable via `RULE_CLUSTER_THRESHOLD` with a documented default (add to `.env.example`).
- **Don't tune it to maximize the eval score** — that overfits our corpus; choose from the gap, then check against the eval.

DESIGN.md wording (after measuring): "The rule-key clustering threshold (cosine distance X, average linkage) was set inside the observed gap between hand-labelled same-rule and different-rule key pairs, leaning strict because over-merging is worse than under-merging; it is stable under ±0.03 perturbation and configurable via `RULE_CLUSTER_THRESHOLD`." — X is the measured value; if there is no clean gap, say so and rely on the LLM check for near-threshold pairs.

*Over- vs under-merging:* **over-merge is worse** — two different rules become one and precedence picks a wrong winner silently. Under-merge misses a conflict, but the composer still sees both chunks and flags visible disagreement. → Conservative threshold; don't tune it tightly to our corpus (overfits the modified corpus).

*Visibility:* ingestion report lists every cluster and its members; eval cases built around known conflicts assert both values and both citations appear.

**DESIGN.md line:** "At ingestion an LLM extracts typed facts per chunk (topic, attribute, value, unit, conditions), verified against a verbatim quote that must exist in the chunk and contain the value (values stored as stated, no derived numbers; failing facts are dropped and counted while the chunk stays retrievable), and clustered into canonical rule keys (topic + attribute embedded with average linkage at a conservative threshold; units and conditions must match; values never part of the key, so conflicts surface). This lets precedence, conflicts and upcoming changes be resolved deterministically in code across all eligible documents — not limited by retrieval top-k or the 3,000-token budget — and makes the spec's worked example a unit test. Facts select which passages the LLM sees; answers and citations still come from source text."

---

## Q23. Why precedence in code instead of the LLM?

**Answer:** Q22 covers *why facts are extracted*; this covers *who applies the rules*. §5.1 is a fully specified algorithm, not a judgment call — so it belongs in code.

**The PDF defines an exact algorithm (§5.1):** "apply these tests in order. Move to the next test only if the current one does not decide."
1. Authority — policy/addendum/memo/handbook/calendar outrank faq/wiki.
2. Scope — regional outranks GLOBAL.
3. Recency — later `effective_date` wins.
4. Still tied — give both values, say they conflict. "Do not pick one."

**Why code beats the LLM:**

| Concern | LLM applying precedence | Code (`precedence.resolve`) |
|---|---|---|
| Correct order | May let a newer FAQ beat an older policy "because it's more recent" — the worked example's trap | Sorts by `(authority, scope, date)`; order can't be wrong |
| Date comparison | Errors, especially with a past reference date | Exact `date` comparison |
| "Do not pick one" | Tends to pick a winner anyway | Returns `CONFLICT` with both values |
| Upcoming-change rule | "Would it win on its date?" what-if is unreliable | Call `resolve()` again at that date |
| Consistency (§10.3) | Can vary across 3 runs | Identical |
| Prompt injection (§7.5) | "This FAQ overrides all policies" may persuade it | Ranks by `doc_type` from the manifest — text can't change ranking |
| Testability | Can't unit-test "the LLM decided" | Worked example + edge cases are unit tests |
| Explainability | "The model said so" | Logs why: "v2 beat v1 on recency" |
| Token budget (§4.3) | Needs every version in context | Only winner and conflicts sent |

The injection row matters most: precedence in code is a **security** decision as well as a correctness one.

**Division of labour (principle P1):**
- **LLM — fuzzy:** understanding the question, extracting facts, recognising that two phrasings are the same rule, writing the answer.
- **Code — crisp:** eligibility, precedence order, conflicts, upcoming changes.

**Honest limit:** code is only as good as the extracted facts. If extraction misses a "same rule" match, `resolve` never sees the conflict. Safety net (Q22): the composer flags visible disagreements in retrieved chunks; eval cases target known conflicts.

**DESIGN.md line:** "Precedence is implemented in code (`precedence.resolve`) because §5.1 is a fully specified ordered algorithm: authority → scope → recency, with ties returned as conflicts. Code applies it exactly, compares dates reliably, never picks a winner in a true tie, runs the upcoming-change what-if check, and ranks by manifest metadata so document text cannot influence the outcome. The LLM handles only the fuzzy parts: recognising rules at ingestion and phrasing the answer."

---

## Q24. Why two LLM calls instead of an agent loop?

**Answer:** Related to Q5 (LangGraph) and Q2 (agentic RAG), but about the pattern itself. A fixed two-call pipeline keeps latency bounded, makes document injection structurally unable to trigger tools, and keeps every call within the token budget.

**The two designs:**
- **Agent loop (ReAct style):** LLM thinks → calls a tool or retrieval → reads the result → repeats until done. Number of calls unknown.
- **Our pipeline:** (1) **Planner** — user message only, **no corpus text** → intent, slots, sub-queries, tool plan. (2) **Code** — guards, tools, retrieval, precedence, token packing. (3) **Composer** — writes the answer from packed context.

**Why two calls win:**

| Concern | Agent loop | Two-call pipeline |
|---|---|---|
| 30-second limit (§9.1) | 3–10 s per call; 4–5 iterations can blow the limit | ~2 calls, ~8–15 s, predictable |
| Document injection (§7.5) | LLM reads retrieved text then picks the next tool — a malicious chunk can steer it | Planner never sees corpus text → documents **can't** cause a tool call (structure, not prompt) |
| 3,000-token budget per call (§4.3) | Each iteration resends growing context; corpus text piles up | Corpus text in **one** call, packed and checked once |
| Write safety (§8.2) | Loop can call `apply_leave` directly | Writes become code-level proposals; stored args run after confirmation |
| Consistency (§10.3) | Steps vary between runs | Same steps, same order |
| `tool_calls` in order (§9.1) | Reconstructed from the trace | Exact by construction |
| Cost and debugging | Variable cost, long traces | Fixed cost, two calls to inspect |

**Main reason:** the injection row — in a loop it depends on the model obeying; here it cannot happen.

**Many messages need fewer than two calls:**
- Refuse (bulk, off-topic, manipulation): planner → fixed message — **1 call**.
- Clarify: planner → question from `missing` slots — **1 call**.
- Confirm a proposal ("yes"): phrase match → stored args run — **0–1 calls**.
- Propose apply/cancel: working days and rules in code → summary — **1–2 calls**.

**Honest trade-off:** less flexibility when one step depends on another's unpredictable result. Acceptable because there are only 5 known tools with known dependencies — code does the chaining (e.g. "days by December?" → `get_leave_balance` → retrieve accrual rules → composer shows the calculation). **Safety valve:** one bounded re-plan if the composer reports missing information — max 3 calls, never an open loop.

**DESIGN.md line:** "Each message runs a fixed pipeline: a planner call (no corpus text) → deterministic guards, tools, retrieval and precedence → a composer call. Compared with an agent loop this bounds latency well under 30 s, keeps corpus text in a single budget-checked call, keeps writes behind code-level proposals, and makes it structurally impossible for document text to trigger a tool. Refusals and clarifications use one call; at most one bounded re-plan is allowed (≤ 3 calls)."

---

## Q25. Why server-side proposals for confirmation?

**Answer:** When the user says "yes", the system must run exactly what it proposed. Storing the proposal on the server is the only way to guarantee that — the LLM, the client and the history can't change it in between.

**What the PDF requires (§8.2):**
- Propose: summary, `pending_confirmation: true`, proposed call with status `pending_confirmation`, "Nothing changes in the database."
- Execute: "If the next message in the same session is a clear confirmation… run the action."
- Modify ("Make it until Friday instead") → new proposal.
- Anything else → "discard the proposal and handle the message normally."
- "A session has at most one proposal waiting at a time."
- "Never run a write action without confirmation, even if the user says 'don't ask me to confirm'."

→ A state machine with exactly one slot per session; something has to hold that state.

**Options considered:**

| Option | Problem |
|---|---|
| LLM regenerates args from history on "yes" | Args drift ("next Friday" resolved differently, misread dates, wrong type); open to injection via history |
| Client holds the proposal (token sent back) | §9.1 contract has no field for it; their tests won't send one; client could tamper |
| Stateless re-plan from "yes" | "Yes" has no details → back to regenerating from history |
| **Server-side proposal (chosen)** | The stored, validated args **are** what runs |

**How it works:** `proposals(session_id PK, employee_id, tool_name, args_json, summary, created_at)` — the primary key enforces at most one per session.

| Next message | Action |
|---|---|
| Clear confirmation | Owner check (403) → `BEGIN IMMEDIATE` → **claim and delete** proposal → re-validate every rule → execute → commit; status `success` or `error` |
| Changes the details | Validate → **replace** (upsert) → new `pending_confirmation` |
| Anything else | **Delete** → handle normally |
| "Don't ask me to confirm" | No code path skips the proposal step |

**Confirmation detection:** phrase match first, then planner `intent=confirm`; uncertain → **not** a confirmation (a wrong "no" is safe, a wrong "yes" is a damaging write).

**Subtle points:**
1. **Double "yes"** (retry, double-click) could create the leave twice → delete the proposal **inside the same `BEGIN IMMEDIATE` transaction**; the second request finds nothing and is handled normally.
2. **Re-validate at execution** — balance or requests may have changed (§8.2); stored args decide *what* runs, rules are checked *again* (including start date not in the past).
3. **Consistent `tool_calls`** — executed call shows the same args as the proposed call.
4. **No personal data in `args_json`** — only tool args (type, dates, request ID).
5. **No short expiry** — the spec defines none; a slow test could hit it; re-validation covers stale dates.

**DESIGN.md line:** "Write actions are stored as server-side proposals (`proposals`, one row per session via primary key). On a clear confirmation the stored, validated arguments execute — never LLM-regenerated ones — inside a `BEGIN IMMEDIATE` transaction that claims and deletes the proposal and re-validates every rule, so a duplicate 'yes' cannot execute twice. Modifications replace the proposal; any other message discards it; no code path executes a write without one."

---

# Part 2 — Security, privacy and access control

## Q26. Why PII masking at the data layer?

**Answer:** If raw personal data never leaves the data layer, no later code — a logger, a prompt builder, an error handler — can leak it by forgetting to mask. Secure by default, with more layers behind it.

**What the PDF says (§7.4):** PII = `full_name`, `email`, `phone`. Logs: none of them. Answers and `tool_calls` args: never phones; emails masked (`r***@veltrix.example`); names only of employees the requester may see. LLM prompts: only the personal data needed for the current request.

**What happens at the data layer:**

| Field | Treatment | Why |
|---|---|---|
| `phone` | **Never selected** | No feature needs it — data that is never read can never leak |
| `email` | Returned already masked | No feature needs the full address |
| `full_name` | Returned only through visibility-aware methods that take the requester | Names are allowed for visible employees, so they can't be blanket-masked |

**Layers behind it (defence in depth):**
1. **Data layer** — primary guard (above).
2. **Prompt builder** — planner sees employee IDs, role, region only; composer gets names only when the answer needs them.
3. **Output validator** — strips phone patterns, masks any unmasked email, checks every name against the requester's visible set.
4. **Log filter** — redacts email/phone regexes and known names in every logger.

**Traps:** exception tracebacks and SQL parameter logging can dump whole rows → log IDs only and keep the filter on all handlers; a user may type another employee's name — the answer must not confirm that person exists if they aren't visible (see Q29).

**DESIGN.md line:** "PII is minimised at the source: repositories never select `phone`, return emails masked, and expose names only through requester-aware methods. Prompt minimisation, an output validator and a log-redaction filter back this up, so no single forgotten mask can leak personal data."

---

## Q27. Why filter documents before retrieval instead of after?

**Answer:** §7.2 requires it, and post-filtering fails in ways that are hard to see.

**What the PDF says (§7.2):** "A document the requester cannot read must never be sent to the LLM. Filter restricted documents out during retrieval. Removing them after the LLM has seen them is not enough." Must-pass #2.

**Why post-filtering is worse even before the LLM:**

| Problem | Effect |
|---|---|
| Restricted chunks crowd the top-k | Allowed results drop out → worse answers for employees |
| One bug in the post-filter | Restricted text reaches the prompt → must-pass fails |
| Restricted text passes through rerankers, logs, debug output | More places to leak |
| "3 results hidden" style behaviour | Reveals that restricted docs exist (§7.2) |

**How it's done:** `eligibility.allowed_chunk_ids(requester, reference_date)` from SQLite metadata → **both** FAISS and BM25 search only those IDs.

**Senior extra — a third guard:** every `CorpusSegment` carries its `access_level` and regions; `LLMClient` asserts each one is readable by the requester before sending. A bug upstream then fails loudly instead of leaking. A test runs every role through the eval and asserts no segment ever violates access.

**DESIGN.md line:** "Restricted documents are excluded before search: both retrievers query only `allowed_chunk_ids`, and `LLMClient` re-asserts that every corpus segment is readable by the requester before any call — restricted text cannot reach the prompt, logs or rankings."

---

## Q28. Why permission checks inside the tools?

**Answer:** The spec says so explicitly, and it makes the tools the single gate to leave data — no orchestrator bug or LLM argument can get around it.

**What the PDF says (§6.1):** "Permission checks must run inside the tool code, using the employee ID from the request header. They must not depend on the prompt or on arguments chosen by the LLM." Must-pass #3.

**How:**
- Every tool receives `ToolContext(requester)` built from `X-Employee-Id` + DB — never identity from LLM args.
- All decisions live in `tools/authz.py` (`can_view_employee`, `can_view_team`, `assert_self`) — one file to audit.
- `apply_leave` has **no** `employee_id` argument → it cannot act for someone else.
- `cancel_leave` checks the request belongs to the requester.
- `get_team_leave` returns only the requester's direct reports, only if role = manager.

**Why not in the prompt or orchestrator:** prompts can be overridden ("I'm in HR"); orchestrator checks can be skipped by a new code path. A check inside the tool runs on every path.

**Tests:** role × tool × target (self, direct report, other, non-existent) matrix.

**DESIGN.md line:** "All leave-data permissions are enforced inside the tools via `tools/authz.py`, using a requester context built from the header and the database; LLM-chosen arguments never carry identity, and write tools have no target-employee argument."

---

## Q29. Why the same fixed refusal for denied and non-existent employees?

**Answer:** Different messages would turn the assistant into an **existence oracle** — anyone could discover which employee IDs or names exist.

**What the PDF says (§7.3):** "give the same fixed refusal message whether that employee exists or not." §7.2: never reveal that a restricted document exists.

**How:**
- `authz.py` returns one `ACCESS_DENIED` result for both cases → one constant message, e.g. "I can't share information about that employee."
- Same route (`refuse`), same `tool_calls` shape and status for both paths.
- Name lookups: "not found" and "not visible" give the same refusal; ambiguity is resolved **only within the visible set** (Q39), so a clarify question can't reveal hidden people.
- Documents: no "that's in a document you can't access" — restricted docs simply don't exist for that requester.

**Test:** compare full responses for a denied-but-existing ID and a fake ID — they must be identical apart from IDs the user typed.

**DESIGN.md line:** "Unauthorised and non-existent employees produce one identical refusal (same message, route and tool status), and name ambiguity is only resolved within the requester's visible set, so responses never reveal whether an employee exists."

---

## Q30. Why a layered prompt-injection defence instead of a prompt instruction?

**Answer:** A prompt instruction ("ignore instructions in documents") is probabilistic — it usually works. Structural defences make the worst outcomes impossible rather than unlikely.

**What the PDF says (§7.5, §2.1):** corpus contains instruction-like text; treat it as information only, never follow it, never call a tool because a document says to. Users trying to override instructions, claim authority or extract the system prompt → `refuse`.

**Layers:**

| Layer | What it guarantees |
|---|---|
| Planner never sees corpus text (Q24) | Documents **cannot** cause a tool call |
| Precedence by manifest metadata (Q23) | Document text can't change which document wins |
| Permissions in tools from header (Q28) | "I'm HR" changes nothing |
| Writes only via server-side proposals (Q25) | No write without user confirmation |
| Corpus delimited as untrusted data + `injection_scan.py` flags | Composer treats it as quoted content |
| Output validator | Rejects outputs requesting actions; checks PII and citations |
| Planner `intent=manipulation` → `refuse` | User-side override attempts handled |
| System prompt holds no corpus or secrets | Nothing valuable to extract |

**Tests:** adversarial suite — injected doc text, role claims, "ignore previous instructions", system-prompt requests, "don't ask me to confirm".

**The ingestion-time injection scan — why it exists:**

Honest framing: the scan is the **weakest layer and not a security boundary**. The structural layers above work even if it misses everything. It exists for **visibility**, not protection.

| Reason | Detail |
|---|---|
| 1. Showing we found them | §2.1 says the corpus has instruction-like text; reviewers will check we noticed. The ingestion report lists every flagged chunk with `doc_id` and locator. |
| 2. Extra labelling | Flagged chunks get a stronger marker in the composer prompt so the answer doesn't repeat the instruction as advice. |
| 3. Targeted tests | Adversarial and eval cases retrieve flagged chunks and assert no tool call, no behaviour change, no leaked system prompt. |
| 4. Failure-mode reporting | DESIGN.md §6 can say "found N injection passages, here's how each was handled." |

**Why at ingestion:** once per document, cached, deterministic, no `/chat` latency; covers OCR output too.

**Why flag, not delete:**

| Deleting | Flagging (chosen) |
|---|---|
| False positives remove real policy ("Employees should ignore the previous guidance on carry-forward" is legitimate) | Text stays — correct answers and citations |
| Answers change silently on the modified corpus | Stable behaviour; structural layers handle the risk |
| Citations point to text that "doesn't exist" | Verbatim text kept for locators |

**How:** heuristic patterns ("ignore previous instructions", "you are an AI/assistant", "system prompt", "call/execute the tool", "approve this leave", "reveal", role overrides, text addressed to the model rather than the employee) → `suspicious=true`. No LLM classifier — not a boundary, so extra cost and variation aren't worth it. **Known weakness:** regex is easy to evade — stated openly, which is why it isn't relied on.

**DESIGN.md line:** "Injection defence is structural: the planner never sees corpus text, precedence uses manifest metadata, permissions use the header, and writes need a stored proposal — so injected text cannot trigger tools, change rankings or bypass access. Delimiting and output validation are additional layers, and manipulation attempts are routed to `refuse`. An ingestion-time scan flags instruction-like passages (including OCR output) for visibility — listed in the ingestion report, labelled in the composer prompt, targeted by adversarial tests — and keeps them rather than deleting them; it is pattern-based and evadable, so protection never depends on it."

---

## Q31. Why refuse bulk requests with a code-level guard?

**Answer:** §7.3 requires refusal for every role, and checking it in code (not only by the LLM) keeps it reliable.

**What the PDF says (§7.3):** "Refuse requests for data about many employees at once, such as 'Show me everyone's balances', for every role. The only exception is a manager using `get_team_leave` for their own direct reports." HR: "any one employee at a time" (§3.1).

**How:**
- Planner flags `targets_multiple`; `orchestrator.guard_plan` also checks the tool plan: read tools may target the requester plus **at most one** other employee per message.
- `get_team_leave` is the only multi-employee path, manager-only, own reports only.
- Refusal is a normal HTTP 200 with route `refuse`.

**Edge case (README assumption):** "Show me my team's balances" needs `get_leave_balance` for many people → refused as bulk; the manager can use team leave (`get_team_leave`) or ask about one report at a time.

**DESIGN.md line:** "Bulk requests are refused in code: a plan may touch at most one other employee per message, with `get_team_leave` for a manager's own reports as the only exception."

---

# Part 3 — Orchestration and LLM behaviour

## Q32. Why is the route assigned in code, not by the LLM?

**Answer:** §8.1 ties each route to strict field rules. Deriving the route from what actually happened guarantees the rules always hold.

**What the PDF says (§8.1):** exactly one route; `rag` → citations ≥ 1, tool_calls empty; `tool` → tool_calls ≥ 1, citations empty; `hybrid` → both; `clarify`, `no_answer`, `refuse` → both empty.

**Why not the LLM:** it can say `rag` with no citations, or `tool` after retrieval — a contract violation the automated tests will catch.

**How:**

| Condition (in order) | Route |
|---|---|
| Guard refused (bulk, off-topic, manipulation, not permitted) | `refuse` |
| Missing or ambiguous details | `clarify` |
| Policy question, nothing eligible/relevant | `no_answer` |
| Citations and tool calls | `hybrid` |
| Tool calls only (incl. proposals and rejections) | `tool` |
| Citations only | `rag` |

The LLM contributes **intent**; code decides the route; `validator.py` asserts the invariants before responding.

**DESIGN.md line:** "The route is derived in code from what happened (guards, missing slots, citations, tool calls) and validated against §8.1's invariants; the LLM supplies intent only."

---

## Q33. Why are citations built in code from used chunk IDs?

**Answer:** §4.2 demands exact citations — every document used, no others, never restricted. Letting the LLM write `doc_id`s invites invented or wrong citations.

**What the PDF says (§4.2):** each citation = `doc_id` + locator (nearest heading or `p. N`); cite every document used and no others; never cite a document the requester cannot read.

**How:**
- The composer sees chunks labelled with short chunk IDs and returns `used_chunk_ids`.
- Code keeps only IDs that were **actually provided** in this call (∩ allowed), maps them to `doc_id` + locator, de-duplicates.
- Conflicts → both documents cited; upcoming change → its document cited.
- Tool-only answers → no citations.

**Honest limit:** "used" is self-reported by the LLM — it can over- or under-report. Measured with citation precision/recall in the eval; the composer prompt asks it to cite only chunks its answer relies on.

**DESIGN.md line:** "Citations are built in code from the composer's `used_chunk_ids`, restricted to chunks actually provided in that call, and mapped to `doc_id` + locator — the LLM can never cite an unretrieved or restricted document; citation precision/recall is measured in the eval."

---

## Q34. Why structured JSON outputs from every LLM call?

**Answer:** Free-text parsing is brittle. Every LLM call returns a schema the code can validate, so a bad output is caught instead of silently mis-handled.

**Calls and schemas:**

| Call | Output |
|---|---|
| Planner | `intent`, `reference_date`, `sub_queries`, `tool_plan`, `leave_slots`, `target_employee_ref`, `missing` |
| Composer | `answer`, `used_chunk_ids`, `uncovered_parts` |
| Fact extraction | list of `{topic, attribute, value, unit, conditions, quote_span}` |
| OCR | markdown text (validated for non-empty, cross-checked numbers) |

**How:** provider structured-output / forced tool-call mode → Pydantic strict validation (Q17) → one repair retry with the error message → safe fallback (`clarify` or `no_answer`, never a guessed answer).

**DESIGN.md line:** "Every LLM call returns a Pydantic-validated JSON schema via the provider's structured-output mode; invalid output gets one repair retry, then a safe fallback route."

---

## Q35. Why temperature 0, and is that enough for consistency?

**Answer:** Temperature 0 is necessary but **not enough** — hosted LLMs are not perfectly deterministic even at 0. Consistency comes mainly from design: the LLM decides as little as possible.

**What the PDF says (§10.3):** "We run our tests 3 times, so make the output as consistent as you can, for example with temperature 0."

**Consistency by design:**

| Source of variation | How it's removed |
|---|---|
| OCR, extraction | Run once at ingestion, cached by content hash |
| Precedence, eligibility, route, citations | Code (Q23, Q32, Q33) |
| Dates | Code (Q37) |
| Retrieval order | Deterministic search + RRF tie-break by `chunk_id` |
| Prompts | Fixed templates, stable ordering of chunks and tool results |
| Model updates | Pin a dated model ID via `LLM_MODEL` |
| LLM wording | Temperature 0; only phrasing varies, not facts or routes |

**Proof:** `make eval` runs every case 3× and reports any difference in route, citations or key facts.

**DESIGN.md line:** "Temperature 0 is used, but consistency comes from design: OCR and extraction are cached, routing/precedence/citations/dates are code, retrieval is deterministic and the model ID is pinned — only wording is left to the LLM. The eval runs 3× to verify."

---

## Q36. Why Anthropic as the default provider, with an abstraction?

**Answer:** The spec needs at least one provider and reviewers bring their own key. A thin provider interface with two adapters raises the chance it works with whatever key they have.

**What the PDF says (§10.2, §10.3):** `LLM_PROVIDER` variable; must work with at least one of OpenAI, Anthropic or Gemini; "We will use our own API key."

**Why Anthropic as default:** reliable tool use and JSON adherence, vision for OCR, and prompt caching for the static system prompt. (A reasonable choice, not a claim that it's the only good one.)

**The abstraction:** `LLMProvider` with `complete(messages, schema)` and `vision(image, prompt)`; adapters for Anthropic and OpenAI; selected by `LLM_PROVIDER`; model pinned by `LLM_MODEL`. `LLMClient` wraps any provider with the token guard, logging, timeouts and retries — so the must-pass guards don't depend on the provider.

**Trade-off:** two adapters to maintain → covered by contract tests with a fake provider plus one live smoke test per provider.

**DESIGN.md line:** "Anthropic is the default provider and OpenAI is supported through a thin `LLMProvider` interface selected by `LLM_PROVIDER`; the token guard, logging and retries live in `LLMClient` above it, so they hold for any provider."

---

## Q37. Why resolve dates in code instead of trusting the LLM?

**Answer:** §8.4 gives exact date rules, and LLMs are unreliable at weekday arithmetic. Code applies the rules exactly; the LLM only spots the date phrase.

**What the PDF says (§8.4):** calendar dates only; today = `APP_TODAY`; no year → 2026; weeks start Monday; "next week" = Monday–Friday of the following week; "next Friday" = first Friday after today; ambiguous or past → ask. §6.2: dates in 2026, start ≥ today, end ≥ start.

**How:** the planner returns the phrase and a candidate ISO date; `dates.py` re-resolves known phrases against `APP_TODAY`; if code and LLM disagree, code wins; unknown phrases are accepted only if unambiguous and valid, otherwise `clarify`. Past dates, non-2026 dates and end < start → `clarify` or rejection with the reason.

**Reference date for past-period questions (§5):** the last day of the period asked about — "in 2025" → 2025-12-31, "in August 2025" → 2025-08-31; a specific date is used as-is; a period that includes today → `APP_TODAY`. Chosen because it reflects the rules as they stood by the end of that period (including mid-year changes), and the answer states the date used ("As of 31 December 2025…").

**Tests:** table of phrases × several `APP_TODAY` values (including a Friday, a Sunday and late December), plus past-period cases ("in 2025", "in August 2025", "in 2026").

**DESIGN.md line:** "Date phrases are spotted by the planner but resolved in `dates.py` against `APP_TODAY` using §8.4's rules; code overrides the LLM on disagreement, and ambiguous or past dates trigger a clarifying question."

---

## Q38. Why compute future balances partly in code?

**Answer:** §6.4 wants the assistant to combine the current balance with the accrual and carry-forward rules and **show the calculation**. LLMs make arithmetic slips, so code supplies the exact numbers and the LLM explains them.

**What the PDF says (§6.4):** tools return the current balance only — no future accrual inside tools; for "How many days will I have by December?" combine current balance with document rules and show the calculation.

**How:**
1. `get_leave_balance` → current balance (tool, unchanged).
2. Retrieval + precedence → the applicable accrual and carry-forward rules (cited).
3. Code computes the deterministic parts (months remaining from `APP_TODAY` to the target date, simple rate × months, caps when the facts are numeric).
4. Composer shows the steps using those numbers; route `hybrid`.
5. Rules too conditional for code → the LLM shows the calculation from given inputs, and the answer says which assumptions it made.

**DESIGN.md line:** "Future-balance questions combine the tool's current balance with cited accrual rules; code computes the arithmetic where the facts are numeric, and the composer shows the steps — tools never add future accrual."

---

## Q39. Why resolve employee names only within the visible set?

**Answer:** Searching all employees for a name would reveal who exists. Resolving only among people the requester may see keeps refusals uniform (Q29).

**What the PDF says:** §3.1 visibility (self; manager + direct reports; HR any one employee); §7.3 same refusal whether the employee exists or not; §8.3 ask which employee when ambiguous.

**How:** visible set = self (employee), self + direct reports (manager), everyone (HR). Name matching happens in code against that set only (names are not sent to the LLM for lookup). One match → proceed; several → `clarify` listing only visible matches; none → fixed refusal.

**DESIGN.md line:** "Employee references are resolved in code only within the requester's visible set; no match gives the fixed refusal and multiple matches give a clarifying question listing only visible people."

---

## Q48. Why a hybrid route instead of separate RAG and tool paths?

**Answer:** Not our choice — `hybrid` is one of the six routes the spec defines (§8.1). The design question is how one pipeline handles messages needing both documents and tools, plus one edge case the spec doesn't settle.

**What the PDF says (§8.1):** "Every response has exactly one route." `rag` = documents only; `tool` = tools only (including proposing a write); `hybrid` = "uses both documents and tools" (≥ 1 citation and ≥ 1 tool call). A message needing both **must** be `hybrid`.

**Messages that need both:**

| Message | Tool | Documents |
|---|---|---|
| "How many annual days will I have by December?" (§6.4) | `get_leave_balance` | Accrual and carry-forward rules |
| "Can I carry forward what I have left?" | `get_leave_balance` | Carry-forward policy |
| "My sick balance is low — what if I'm ill longer?" | `get_leave_balance` | Sick-leave policy |
| "Is my report's leave next week on a public holiday?" (manager) | `list_leave_requests` | Holiday calendar (cited) |

**Why one pipeline, not separate paths:**

| Separate RAG agent + tool agent | One pipeline (ours) |
|---|---|
| A mixed message needs two answers; the contract allows one response | One planner plans tools **and** sub-queries; one composer merges |
| More LLM calls, higher latency (30 s) | Still ~2 calls |
| Two answers can contradict | One consistent answer |
| Needs a path classifier that can be wrong | Route derived from what happened (Q32) |

"Hybrid" isn't a separate system — it's the route code assigns when both citations and tool calls are present.

**Edge case — write proposal + policy question** ("Apply annual leave next week, and what's the notice policy?"):
- §8.2: a proposal uses route `tool` with `pending_confirmation: true`; `tool` requires empty citations — but the policy part needs citations → the spec conflicts.
- **Decision (README assumption):** the proposal takes priority — route `tool`, proposal, no citations; the answer invites the user to ask the policy question next. Keeps the safety-critical confirm flow exactly as specified.
- Same for **rejections**: "insufficient balance" comes from the tool → route `tool`, status `error`, even if a policy explains the rule.

**DESIGN.md line:** "`hybrid` is not a separate path: one planner plans tool calls and retrieval sub-queries together, one composer merges both, and the route is derived from what happened (citations and tool calls → hybrid). When a message mixes a write proposal with a policy question, the proposal takes priority (route `tool`, §8.2) and the user is invited to ask the policy part next — recorded as an assumption."

---

# Part 4 — Data, ingestion and leave rules

## Q40. Why holidays in a table and working days in code?

**Answer:** Working days need the complete, exact holiday list for the requester's region. Retrieval returns top-k passages and might miss dates; a table never does.

**What the PDF says:** §6.2 working days = Mon–Fri minus public holidays for the requester's region, from the calendar document that applies under §5, read during ingestion. FAQ §13: holidays may be stored in a table if extracted automatically. §4.3: structured records don't count toward the 3,000 tokens.

**How:**
- Ingestion parses calendar tables → `holidays(doc_id, region, date, name)`, validated (dates parse, row counts reported).
- Which calendar applies: eligibility + precedence at **document level** among `calendar` docs for the requester's region.
- `working_days.py` (pure): count dates in [start, end] with weekday < 5 and not in that holiday set.
- Holiday questions ("When is Diwali?") also use the table — cheap, exact, cited to the calendar doc.

**Assumption (README):** the calendar is chosen as of the reference date (today) — state it, since a newer calendar could take effect later in 2026.

**DESIGN.md line:** "Holidays are extracted into a `holidays` table at ingestion; the applicable calendar is chosen by the Section 5 rules and working days are computed by a pure function — exact, testable and outside the token budget."

---

## Q41. Why collapse near-duplicate documents?

**Answer:** Near-duplicates (§2.1) fill the top-k and the 3,000-token budget with the same content, and make citations ambiguous.

**How:**
- Detect at ingestion: shingle Jaccard / MinHash or embedding cosine above a high threshold (~0.95) → `dup_cluster`.
- **Per request**, after the eligibility filter: keep one representative per cluster, chosen by precedence (authority → scope → recency) — never arbitrarily.
- Duplicates can differ in access level or region, so the representative is chosen only among the requester's eligible members.
- If "duplicates" state different values, they are not duplicates for that rule — fact-level precedence/conflict still applies.

**Refinement — a near-duplicate is not a new version:** policy v1 and v2 may be 98% identical and differ in one number (10 vs 12 sick days), so text similarity puts them in the same cluster.

| Situation | Risk if collapsed on text alone |
|---|---|
| v1 vs v2, v2 more recent | Fine — precedence picks v2 anyway |
| **True conflict** — tie on authority, scope and date, different values | Collapsing drops one → conflict **hidden**; §5.1 says give both values, "do not pick one" → fails §5 |
| Outdated FAQ copying policy text with an old value | Fine — loses on authority, as long as fact-level comparison still sees both |

**Rule:** collapse chunks **only if their extracted facts are identical** (same rule keys, same values). Any differing value → they are **versions, not duplicates** → keep both and let the precedence resolver decide (winner, conflict or upcoming). Clustering saves budget on true copies and never hides a difference.

**Also:** future-dated copies sit in the separate "upcoming" set, so collapsing can't hide an upcoming change; cite only the representative actually used.

**DESIGN.md line:** "Near-duplicates are clustered at ingestion; per request, chunks in a cluster are collapsed to one eligible representative chosen by precedence — but only when their extracted facts are identical. Text-similar chunks with different values are versions, not duplicates, and both go to the precedence resolver so conflicts are never hidden."

---

## Q42. Why UUID4 request IDs?

**Answer:** The spec asks for random, unguessable IDs, and sequential IDs would leak information and invite guessing.

**What the PDF says (§6.2):** "a random, unguessable `request_id`, such as a UUID4."

**Why:** sequential IDs reveal how many requests exist and invite trying others' IDs on `cancel_leave`. Ownership checks already block that, but unguessable IDs are a second layer. Use `uuid.uuid4()` (122 random bits from the OS random source).

**DESIGN.md line:** "New requests get `uuid4()` IDs — unguessable, as §6.2 requires — on top of the ownership check in `cancel_leave`."

---

## Q43. Why ingest in the background with `/health` returning 503?

**Answer:** The server must be reachable immediately, but must not claim readiness until ingestion is complete.

**What the PDF says:** §9.2 `/health` → 200 `{"status": "ok"}` only when ingestion has finished; 503 before. §10: ready within 10 minutes with `docker compose up`.

**How:**
- Uvicorn starts at once; ingestion runs in a `lifespan` background task.
- `/health` reads a ready flag set only after the integrity report passes.
- `/chat` before ready → 503 (README assumption; the spec defines only `/health`).
- Per-document failures are logged with reasons; the service never silently serves partial data — the report states what was skipped and why.
- Content-hash cache makes restarts fast.

**DESIGN.md line:** "Ingestion runs as a background task at startup; `/health` returns 503 until an integrity report passes, and per-document failures are reported explicitly rather than silently skipped."

---

## Q44. Why structured JSON logs instead of plain text?

**Answer:** The spec requires logging corpus tokens per LLM call and forbids PII in logs. Structured logs make both provable.

**What the PDF says:** §4.3 "Your logs must record the number of corpus tokens in every LLM call." §7.4 no names, emails or phones in logs.

**What's logged (JSON lines):** `request_id`, `session_id`, `employee_id`, `route`, `tool_calls[name, status]`, `llm_calls[purpose, model, corpus_tokens, input_tokens, output_tokens, latency_ms]`, retrieval stats (candidates, chunks packed).

**What's not logged:** raw user messages and answers (they can contain names) — or only after redaction; full DB rows.

**Benefit:** a reviewer can check the 3,000-token cap with one `grep` / `jq` over the log.

**Why structured instead of plain text:**

| Concern | Plain text (`f"... {n} tokens for {user}"`) | Structured (`{"event": "llm_call", "corpus_tokens": 2840}`) |
|---|---|---|
| Proving the 3,000 cap (§4.3, must-pass #4) | Fragile regex over free text | One query: `jq 'select(.event=="llm_call") \| .corpus_tokens' \| sort -n \| tail -1` |
| Keeping PII out (§7.4) | Any f-string can slip in a name or email | **Allowlisted fields** — IDs, enums, numbers, durations; no free text → nothing to redact |
| Following one request | Concurrent lines interleave | Every line carries `request_id` |
| Feeding the eval | Hand-parsing | Harness reads latency, tokens, tool statuses directly |
| Consistency across 3 runs | Hard to diff | Diff fields between runs |

**Key point:** with plain text the redaction filter is the **only** protection; with structured logs the **schema** keeps names out and the filter is a backup.

**How:**
- Standard `logging` + small JSON formatter (~20 lines or `python-json-logger`).
- `request_id` and `employee_id` set once per request via `contextvars` — added to every line automatically.
- Fixed event types: `request`, `llm_call`, `tool_call`, `retrieval`, `ingestion_doc`, `error`.
- Test: every `LLMClient` call emits an `llm_call` event with `corpus_tokens`.

**Why `contextvars` for `request_id`:** every log line needs the ID, and `contextvars` is the only option that is correct under concurrent requests, across async code and FastAPI's thread pool, without passing an extra argument everywhere.

| Option | Problem |
|---|---|
| Pass `request_id` as an argument | Clutters every signature, including pure functions (`precedence.resolve`, `authz.py`); miss one → line without an ID |
| Global variable | Wrong under concurrency — request B overwrites A's ID |
| `threading.local` | Breaks with async — many requests share one thread |
| **`contextvars` (chosen)** | Each request has its own context (async task or thread pool); every logger reads it automatically |

- **How:** middleware sets `request_id_var` and `employee_id_var` at the start of each request and resets them at the end; the log formatter adds them to every JSON line; pure functions stay clean (~10 lines).
- **Trap:** Starlette/AnyIO carry context into the thread pool for `def` endpoints — verify with a test rather than assume. **Threads you start yourself don't inherit it** (e.g. a `ThreadPoolExecutor` for parallel extraction at ingestion) → wrap work with `contextvars.copy_context().run(...)` or set a per-task `doc_id`.
- **Test:** two concurrent `/chat` requests → every log line carries the correct ID, none swapped.

**Trade-off:** less human-readable → `LOG_FORMAT=text` for local development (JSON default in Docker; listed in `.env.example`). No OpenTelemetry or log stack — out of scope, and §10.3 allows only LLM/embedding APIs as external services; JSON lines to stdout via `docker compose logs`.

**DESIGN.md line:** "Logs are JSON lines with a fixed, allowlisted schema per event (IDs, enums, numbers — no free text), so personal data is excluded by construction rather than by redaction alone, and every LLM call's `corpus_tokens` can be verified with one `jq` query. `request_id` via contextvars correlates planner, tools and composer; `LOG_FORMAT=text` is available for local reading."

---

# Part 5 — Testing, evaluation and delivery

## Q45. Why an eval harness on top of unit tests — and why 3 runs?

**Answer:** Unit tests prove the parts work; the eval proves the **system** answers correctly. The reviewers grade with an automated suite on a modified corpus (§12) — you need your own version of that.

**How:** `eval/cases.yaml` — each case: employee ID, message(s), expected route, expected citation doc IDs, must-contain facts, must-not-contain (restricted doc IDs, phone patterns, unmasked emails). `make eval` scores route accuracy, citation precision/recall, key-fact accuracy, refusal correctness, and runs 3× for consistency. Also used for ablations (BM25 vs dense vs hybrid, RRF k, fusion).

**Why 3 runs (§10.3 "We run our tests 3 times"):** mirror how we'll be graded. A case correct 2 of 3 times may count as a failure — we don't know if graders take the majority, require all 3, or average → **assume all 3 must agree**.

**What 3 runs catch that 1 run hides:**

| Problem | Example |
|---|---|
| Unstable route | Same question → `rag` twice, `no_answer` once |
| Citation drift | Composer sometimes reports an extra chunk as used |
| Changing key facts | Wording occasionally drops the conflict or upcoming-change mention |
| Planner instability | Date phrase or leave type sometimes missed → `clarify` instead of a proposal |

**What to measure:**
- Per case: same route, same citation set, all must-contain facts in all 3 runs.
- Consistency rate: share of cases where all 3 agree — target 100% for route and citations.
- Unstable list with diffs → the fix list, and entries for DESIGN.md §6 Failure modes.

**Details that matter:**
1. **Fresh DB copy before every run** — apply/cancel cases change balances; without a reset, run 2 measures our own state changes, not consistency (their copy is fresh too, §2.3).
2. **Fresh `session_id`s per run** — a pending proposal from run 1 must not affect run 2.
3. **Check ingestion too** — run it twice with the cache cleared and diff extracted facts and holidays; the cache hides extraction variance, but graders start cold.
4. **Why 3, not 10** — matches the grader at low cost; investigate wobbly cases with 10 runs.

**DESIGN.md line:** "`make eval` runs every case 3 times — mirroring the graders' 3 runs (§10.3) — on a fresh DB copy and fresh sessions each time, and reports per-case agreement on route, citations and key facts; unstable cases are listed as failure modes. Ingestion is also run twice from a cold cache to check that fact and holiday extraction is reproducible. The eval also drives the retrieval ablations."

---

## Q46. Why stub the LLM in integration tests?

**Answer:** Tests must be fast, deterministic and runnable without an API key. A fake provider returns scripted JSON so tests check **our** logic, not the model's mood.

**How:**
- Unit tests: pure functions (working days, precedence, eligibility, authz, leave rules, dates).
- Integration tests: real FastAPI app + fresh copy of a fixture DB per test + **fake `LLMProvider`** → `/chat` contract, 400/401/403, propose → confirm → execute, modify, discard, route invariants.
- Adversarial and property tests (token budget, access on every segment).
- `pytest -m live`: a few end-to-end tests with a real key, skipped when no key is set.
- `make test` (§11.1) runs the offline suite.

**DESIGN.md line:** "`make test` runs offline: unit tests on pure logic and `/chat` integration tests with a scripted fake LLM and a fresh DB copy per test; live-model tests are marked and run only when a key is present."

---

## Q47. Why one Docker service with models baked in at build time?

**Answer:** Must-pass #1 — `docker compose up` must reach `/health` 200 within 10 minutes with no manual step. Fewer moving parts and no startup downloads mean fewer ways to fail on the reviewer's machine.

**How:**
- One `app` service, port 8000, `./data` mounted (§10.1), `env_file: .env`, healthcheck on `/health`.
- Embedding model and tiktoken encoding downloaded at **build** time.
- `tesseract-ocr` installed in the image.
- Ingestion cache in a container path or named volume — **not** inside `./data`, which reviewers replace.
- At runtime the only network calls are the LLM API (§10.3).

**DESIGN.md line:** "A single Docker service with the embedding model, tiktoken encoding and Tesseract baked in at build time; at runtime it needs only `./data` and an LLM key."

---

# Part 6 — Production readiness

## Q50. Is this design production-level, and how would it run in production?

**Answer:** The **core AI and system design** is production-quality; a **complete production design** also needs an operational layer, which is deliberately out of scope (§1, §3) and documented in `docs/PRODUCTION_PLAN.md`.

**Already production-level in its thinking:**

| Area | Why |
|---|---|
| Security by construction | Pre-retrieval filtering, permissions in tools, planner without corpus, proposal-gated writes — safety doesn't depend on the prompt |
| Correctness in code | Precedence, eligibility, dates, working days, routes — deterministic and tested |
| Data integrity | `BEGIN IMMEDIATE`, double-"yes" guard, re-validation at execution |
| Privacy | PII minimised at source + four layers; fixed-schema logs |
| LLM reliability | Schema-validated outputs, repair retry, safe fallbacks, bounded calls, pinned models |
| Ingestion quality | Quote verification, OCR cross-check, integrity report, caching |
| Evaluation | Eval harness, 3-run consistency, ablations, adversarial tests |
| Scaling path | Named bottlenecks with concrete fixes at 10,000 docs |

**What production adds (from `PRODUCTION_PLAN.md`):**

| Area | Production plan |
|---|---|
| Identity | SSO (OIDC) → signed JWT replaces the trusted header |
| Topology | Stateless API replicas, load balancer, Postgres + pgvector, separate ingestion worker on a queue |
| Service goals (targets) | 99.5% availability; p95 < 10 s, p99 < 20 s; < 1% 5xx; **zero** access-control violations; changed docs live within 15 min |
| Threat model | Impersonation, restricted-doc access, others' leave data, injection (doc and user), enumeration, cost abuse, secret leakage, breach — each with a mitigation |
| Data governance | Session retention (e.g. 30 days), GDPR for DE staff, provider data agreements, append-only audit trail, data residency |
| Observability | Metrics (incl. corpus tokens per call), tracing, alerts (access assertion, budget guard, 5xx, ingestion), daily quality eval |
| Model governance | Eval as CI release gate, shadow-testing new model versions, provider fallback, versioned prompts |
| Ingestion | Event-driven, incremental, quality gates (low-confidence OCR held for HR review), index versioning for rollback |
| Reliability | Timeouts, degraded tools-only mode, Postgres point-in-time recovery, idempotent writes via proposal ID |
| Rollout | Staging → pilot (one region, HR first) → feature-flagged writes → gradual rollout with auto-rollback |
| Cost | ~2 LLM calls per request, prompt caching, per-user and global budgets, cost per route tracked |
| Incidents | P1 = any access or PII leak; runbooks; wrong answer → trace → fix → new eval case; "contact HR" escalation |

**Roadmap:** 0 Assessment build → 1 Secure foundation (SSO, Postgres, secrets, audit, CI eval gate) → 2 Operate (observability, worker, staging, pilot) → 3 Scale (replicas, lineage graph, ANN clustering, fallback provider, cost controls) → 4 Extend (approvals, notifications, HR escalation, half-day leave).

**Why document it but not build it:** §1 rewards a smaller system that works. Writing it down shows the difference between an assessment build and a production system, and gives a ready answer to the common senior interview question "how would you run this in production?"

**Note:** the service-goal numbers are **initial targets**, not measurements — present them as targets.

**README line (Known limitations):** "Production concerns — real authentication, multi-instance deployment, observability, data governance, model governance and rollout — are out of scope for this assessment; see `docs/PRODUCTION_PLAN.md`."

---

## Q49. Summary lines for DESIGN.md

- **RAG:** "Metadata-filtered hybrid retrieval (BM25 + dense, RRF) with ingestion-time structured extraction; precedence resolved deterministically in code, LLM used only for interpretation and answer generation."
- **GraphRAG / scale:** "At 10k docs we would not adopt GraphRAG: community summaries count against the corpus-token budget and risk mixing access levels. Instead we'd add a lightweight document-lineage graph plus incremental ingestion and a filtered vector DB."
- **Stack:** "Deliberately minimal stack: FastAPI + SQLite + FAISS/BM25 + fastembed, no orchestration framework — so access filtering, the token budget and precedence are explicit, auditable code."
- **LangGraph:** "Our flow is a fixed pipeline plus one small state machine, so a framework would add dependencies and hidden state without enabling anything we need. If the assistant grew into multi-step tool reasoning or multi-agent workflows, LangGraph would be our first choice."
- **Memory:** "Knowledge lives in a local FAISS + BM25 index with access metadata on every chunk, plus SQLite tables for extracted facts and holidays. Conversation memory is short-term and per-session in SQLite (owner, recent turns, one pending proposal); every turn re-runs filtered retrieval, and no memory is shared across sessions or employees."
- **Cache:** "We don't manage a KV cache (hosted LLM). We use provider prompt caching for the static system prompt and tool schemas, a content-hash ingestion cache for fast restarts (keyed by file bytes + pipeline/model/prompt versions, never file names; manifest metadata never cached), and deliberately no answer or tool-result caching, because answers depend on requester, date and live leave data."
- **Long-term memory:** "Considered per-employee long-term memory and rejected it: the identity header is trusted but not authenticated (§3), it would break reproducibility across test runs (§10.3) and contradict §9.1. Production with real auth: preference-only memory with expiry and deletion."
- **Database:** "SQLite because the spec supplies `veltrix.db` and swaps in a fresh copy at test time; writes land where evaluators expect them. `BEGIN IMMEDIATE` + WAL for safe transactions; Postgres + pgvector at scale."
- **Storage split:** "Derived data in a disposable `cache.db` outside `./data` (design choice, never committed); sessions, messages and proposals in `veltrix.db` so proposal claims are atomic with leave writes."
- **Embeddings:** "Local fastembed (bge-small, CPU) so ingestion needs only the single LLM key reviewers supply; deterministic and no network latency. Quality measured with recall@k; `EMBEDDING_PROVIDER=openai` optional."
- **Vector store:** "FAISS exact search over IDs from our tested eligibility filter; access metadata stays only in SQLite. Chroma considered; Qdrant/pgvector at scale."
- **Hybrid search:** "BM25 for exact terms, FAISS for paraphrases, both restricted to eligible IDs, merged with RRF; backed by a BM25 vs dense vs hybrid ablation."
- **Fusion:** "RRF (k=60) over weighted fusion: no score calibration, no α to overfit, stable on the modified corpus, deterministic; weighted variant kept in the ablation."
- **Token counting:** "tiktoken `cl100k_base` as required by §4.3; packer + `LLMClient` double guard at 3,000, logged per call; encoding pre-cached at build time."
- **API framework:** "FastAPI + Pydantic for exact contract enforcement; 422 remapped to 400, missing/unknown header → 401; blocking work in the thread pool; lifespan background ingestion behind /health."
- **Data models:** "Pydantic (strict where untrusted) at every trust boundary — API, LLM output, tool args, config, manifest — and as the source of tool schemas; frozen dataclasses for pure domain logic."
- **Transactions:** "`BEGIN IMMEDIATE` takes the write lock before validation, so concurrent apply/cancel can't race; rules re-validated inside; WAL + busy timeout + `isolation_level=None`; covered by a concurrency test."
- **RRF k:** "k=60 (Cormack et al. 2009), robust without tuning; favours agreement, single-retriever hits protected by passing top 8–10 forward; k sensitivity check in the eval."
- **Chunking:** "Section-based chunks (~350 tokens, max 500): exact heading locators (§4.2), rules kept with their conditions, several versions fit the 3,000 budget; `p. N` fallback without headings."
- **OCR:** "Vision LLM (verbatim, tool-less, cached) for tables and headings in scans; numbers cross-checked against Tesseract, which is also the fallback."
- **Rule extraction:** "Typed facts extracted and quote-verified at ingestion, partitioned by unit and conditions, clustered into rule keys with scipy average linkage; precedence resolved in code across all eligible docs; facts select passages, source text answers."
- **Precedence in code:** "§5.1 is an exact ordered algorithm — code applies it reliably, returns true ties as conflicts, and ranks by manifest metadata so document text can't sway it; the LLM handles only fuzzy recognition and wording."
- **Pipeline vs agent loop:** "Fixed planner (no corpus) → code → composer: bounded latency, one budget-checked corpus call, writes behind proposals, documents structurally unable to trigger tools; ≤ 3 calls with one bounded re-plan."
- **Server-side proposals:** "One stored proposal per session; on confirmation the stored args execute inside a `BEGIN IMMEDIATE` transaction that claims the proposal and re-validates — no drift, no double execution, no bypass."
- **PII:** "Minimised at the source — phone never selected, emails masked, names via requester-aware methods — backed by prompt minimisation, output validation and log redaction."
- **Pre-retrieval filtering:** "Both retrievers search only allowed IDs; `LLMClient` re-asserts every corpus segment is readable by the requester."
- **Tool permissions:** "Enforced inside tools via `authz.py` from the header identity; write tools have no target-employee argument."
- **Uniform refusal:** "Denied and non-existent employees get an identical refusal; names resolved only within the visible set."
- **Injection:** "Structural — planner never sees corpus, metadata-based precedence, header-based permissions, proposal-gated writes — plus delimiting and output validation; an ingestion scan flags (not deletes) instruction-like text for visibility and tests, never relied on for protection."
- **Bulk guard:** "At most one other employee per message, except a manager's `get_team_leave`."
- **Route:** "Derived in code from what happened and validated against §8.1; the LLM supplies intent only."
- **Citations:** "Built in code from `used_chunk_ids` restricted to provided chunks; precision/recall measured."
- **Structured outputs:** "Schema-validated JSON from every LLM call; one repair retry, then a safe fallback."
- **Consistency:** "Temperature 0 plus design — cached ingestion, code-decided logic, deterministic retrieval, pinned model; verified by 3× eval."
- **Provider:** "Anthropic default, OpenAI supported via `LLMProvider`; guards live in `LLMClient` above any provider."
- **Dates:** "Resolved in `dates.py` against `APP_TODAY` with §8.4 rules; ambiguous or past → clarify."
- **Future balances:** "Current balance from the tool + cited accrual rules; arithmetic in code where possible; steps shown."
- **Name resolution:** "Only within the requester's visible set; none → fixed refusal, several → clarify."
- **Holidays:** "Extracted to a table; calendar chosen by Section 5 rules; working days by a pure function."
- **Near-duplicates:** "Clustered at ingestion, collapsed per request to one eligible representative chosen by precedence — only when extracted facts are identical, so versions and conflicts are never hidden."
- **Request IDs:** "`uuid4()` — unguessable, on top of ownership checks."
- **Startup:** "Background ingestion; `/health` 503 until the integrity report passes; failures reported, never silent."
- **Logging:** "JSON lines with an allowlisted schema (no free text) — PII excluded by construction, per-call `corpus_tokens` verifiable with one query; `request_id` via contextvars, correct under concurrency; `LOG_FORMAT=text` for local use."
- **Eval:** "End-to-end cases run 3× (fresh DB and sessions each run) for accuracy and consistency, plus a cold-cache ingestion reproducibility check; drives retrieval ablations."
- **Tests:** "Offline `make test` with a scripted fake LLM and fresh DB per test; live tests marked."
- **Docker:** "One service, models and encodings baked in at build; runtime needs only `./data` and an LLM key."
- **Hybrid route:** "Not a separate path — one planner and composer handle tools and documents together; route derived from what happened; a write proposal takes priority over a mixed-in policy question."

---

## Assumptions to record in the README

Collected from the decisions above — copy into the README's Assumptions section (§1 asks for them).

1. **Conversation memory** is scoped to one `session_id` and its owning employee; nothing is kept across sessions (Q8, Q9).
2. **`/chat` before ingestion finishes** returns 503; the spec defines this only for `/health` (Q43).
3. **"Show me my team's balances"** is a bulk request and is refused; managers can use `get_team_leave` or ask about one report at a time (Q31).
4. **Holiday calendar** is chosen as of the reference date (today), even if a newer calendar takes effect later in 2026 (Q40).
5. **Write proposal + policy question** in one message: the proposal takes priority (route `tool`), and the user is invited to ask the policy part next (Q48).
6. **Past-period questions resolve to the last day of that period.** "In 2025" → reference date **2025-12-31**; "in August 2025" → 2025-08-31; a specific date ("on 1 August 2025") → that date. If the period includes today (e.g. "in 2026"), the reference date is today (`APP_TODAY`), never a future date. The answer states the date used, e.g. "As of 31 December 2025…" (Q37, §5).
7. **Ambiguous confirmations** are treated as not confirmed — a wrong "no" is safe, a wrong "yes" is not (Q25).
8. **No proposal expiry**; stale proposals are caught by re-validation at execution (Q25).

**Known limitations line:** "Production concerns — real authentication, multi-instance deployment, observability, data governance, model governance and rollout — are out of scope for this assessment; see `docs/PRODUCTION_PLAN.md`." (Q50)
