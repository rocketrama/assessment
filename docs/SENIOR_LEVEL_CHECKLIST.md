# Senior AI Engineer — Assessment Checklist
**Project:** Veltrix PeopleOps Assistant (HR RAG + leave tools API)
**Purpose:** Self-review checklist to make sure the submission demonstrates senior-level engineering, not just a working demo.

---

## Level 1 — Must-pass (baseline; failure = review stops)

- [ ] `docker compose up` starts the service on port 8000; `/health` → 200 within 10 min (503 until ingestion done)
- [ ] Restricted documents are filtered **before retrieval** — never sent to the LLM
- [ ] No employee can see leave data they are not allowed to see (checks run **inside tool code**, using `X-Employee-Id`)
- [ ] Every LLM call ≤ **3,000 corpus tokens** (`tiktoken` `cl100k_base`), enforced in code and logged per call
- [ ] Works on a **modified corpus** — no hardcoded facts, file names, values or layouts

## Level 2 — Required features (correctness)

### Ingestion & retrieval
- [ ] All formats ingested automatically: PDF, DOCX, Markdown, HTML
- [ ] Tables parsed (holiday calendars, accrual rates)
- [ ] Scanned document read via OCR / vision model
- [ ] Near-duplicates handled; unrelated docs don't pollute answers
- [ ] Citations: `doc_id` + locator (section heading or `p. N`); cite all used, none unused, never restricted

### Document applicability (Section 5)
- [ ] Eligibility: access level + region (or GLOBAL) + effective_date ≤ reference date
- [ ] Reference date = `APP_TODAY`, or the past date the user asks about
- [ ] Precedence in code: **Authority → Scope → Recency**
- [ ] Unresolved conflict → show both values, cite both, say they conflict
- [ ] Upcoming changes mentioned (if readable, same region, and would win on its date)
- [ ] Partial answers state what is not covered; nothing eligible → `no_answer`

### Tools & leave rules
- [ ] Exact tool + argument names: `get_leave_balance`, `list_leave_requests`, `get_team_leave`, `apply_leave`, `cancel_leave`
- [ ] Working days = Mon–Fri minus region holidays **extracted at ingestion**
- [ ] `apply_leave` rejects: zero working days, insufficient balance, overlap with own pending/approved, dates outside 2026 or in the past
- [ ] Never assume leave type — ask
- [ ] Apply/cancel each in **one DB transaction**; UUID4 request IDs
- [ ] Cancel only pending/approved with start_date after today; refund days
- [ ] Rejections give the specific reason; tool status `error`
- [ ] Future-balance questions: current balance + doc accrual rules, calculation shown

### Conversation behaviour
- [ ] Exactly one route per response: `rag | tool | hybrid | clarify | no_answer | refuse`
- [ ] Two-step write: propose (`pending_confirmation: true`) → execute on clear confirmation
- [ ] Changed details → new proposal; anything else → discard proposal; max one pending per session
- [ ] Never execute writes without confirmation (even if user says "don't ask")
- [ ] Clarify asks for **everything missing in one message**
- [ ] Date handling: no year = 2026, weeks start Monday, "next week"/"next Friday" rules, ask if ambiguous or past
- [ ] Off-topic → polite refuse + what the assistant can help with

### Identity, access & privacy
- [ ] Identity/role/region only from DB; ignore claims in messages
- [ ] Session owned by first sender; other employee → **HTTP 403**
- [ ] Never reveal restricted docs exist (no title, id, or "you can't access that")
- [ ] Bulk data requests refused for every role (except manager `get_team_leave` for direct reports)
- [ ] **Same fixed refusal** whether the employee exists or not
- [ ] No names/emails/phones in logs; phones never shown; emails masked `r***@domain`
- [ ] Only minimum PII sent to LLM
- [ ] Prompt injection in docs ignored; manipulation / system-prompt requests → `refuse`

### API contract
- [ ] `POST /chat` request/response shape exactly as spec; `tool_calls` list every call in order
- [ ] 400 invalid body, 401 missing/unknown employee, 403 session mismatch; refusals are HTTP 200
- [ ] Each request < 30 seconds
- [ ] Temperature 0 / deterministic outputs (tests run 3×)

### Deliverables
- [ ] Clean module separation: ingestion / retrieval / orchestration / tools / data access
- [ ] `Dockerfile`, `docker-compose.yml` (mounts `./data`), `.env.example` (no real keys)
- [ ] `make test` runs pytest: working days, precedence, access control per role, apply/cancel rules, `/chat` integration
- [ ] `README.md`: setup, architecture diagram (Mermaid), Assumptions, Known limitations, sample queries
- [ ] `DESIGN.md` (≤ 2 pages) covering all 7 required topics

## Level 3 — Senior differentiators (what makes it stand out)

### Design & reasoning
- [ ] **Rule extraction at ingestion** — structured rules (topic, value, unit, region, doc_id) so "same rule" conflicts are detected and resolved in code, testable
- [ ] Deterministic logic in code; LLM only for understanding, routing and wording
- [ ] Explicit trade-offs in DESIGN.md: what was chosen, what was rejected, why, and the cost
- [ ] Assumptions written with reasoning, not just listed
- [ ] Concrete 10,000-doc analysis: first bottleneck named (ingestion time, O(n²) conflict detection, token budget vs. more conflicting sources, re-index cost) + fix for each

### Evaluation
- [ ] **Eval harness** (`make eval`): question set with expected route, citations, key answer facts → scorecard (route accuracy, citation precision/recall, refusal correctness)
- [ ] Consistency check: eval run 3× produces same results
- [ ] **Modified-corpus test**: change a value, re-ingest, assert the answer changes
- [ ] Retrieval quality measured (recall@k on the eval set), not assumed

### Security (defence in depth)
- [ ] Layered: pre-retrieval filtering + in-tool permission checks + schema-validated LLM output + PII masking at output boundary
- [ ] Adversarial test suite: injection in docs, "I'm HR" claims, bulk data, employee-existence probing, session hijack, system-prompt extraction, "don't confirm" bypass
- [ ] Tool args from LLM are never trusted for identity (always the header)

### Reliability & production readiness
- [ ] Handles malformed LLM output (validation + retry/fallback)
- [ ] Handles LLM timeout within the 30 s budget
- [ ] Partial ingestion failure logged clearly; service doesn't silently serve incomplete data
- [ ] Structured logging (request_id, employee_id, route, tool calls, corpus tokens, latency) — PII-free
- [ ] LLM provider swappable behind an interface
- [ ] Typed code (type hints, Pydantic models), linted, readable

### Honesty & communication
- [ ] Failure modes section: what was found, fixed / not fixed, why
- [ ] Known limitations stated plainly (reviewers reward honesty)
- [ ] Clean, small, logical commit history that tells the build story
- [ ] README lets a reviewer run and verify everything in minutes

---

## Priority order if time runs short
1. Level 1 must-pass items
2. Access control + PII + tool rules (Level 2)
3. Precedence engine + citations (Level 2)
4. Tests required by the spec
5. DESIGN.md + README
6. Eval harness + adversarial tests (Level 3)
7. Remaining polish

## Suggested 7-day plan
| Day | Focus |
|---|---|
| 1–2 | Ingestion (formats, tables, OCR, rule/holiday extraction), data layer, tools |
| 3 | Retrieval with access filtering, precedence engine |
| 4 | Orchestration: routes, clarify, propose/confirm state machine |
| 5 | Tests: unit, integration, adversarial |
| 6 | Eval harness, modified-corpus test, failure analysis |
| 7 | DESIGN.md, README, cleanup, final run on fresh `./data` |

---

## What separates senior from mid-level
| Area | Mid-level | Senior |
|---|---|---|
| Ambiguity | Asks or guesses quietly | Writes explicit assumptions with reasoning, e.g. "an FAQ that contradicts policy is treated as outdated because..." |
| "Same rule" detection | Leaves it to the LLM | Extracts structured rules (topic, value, unit, region) at ingestion, so conflicts are found in code and can be tested |
| Modified corpus | Works on the given files | Designed to generalize, with no assumptions about file names, headings or table layouts; tested by changing the corpus yourself |
| Evaluation | Unit tests only | Builds a small eval harness: questions with expected route, citations and answer, scored automatically and run three times to check consistency |
| Failure analysis | "It works" | Lists the real failures found (an OCR miss, a table parse error, a wrong route) with fixed/not fixed and why |
| Security | Prompt says "don't follow instructions" | Defence in layers: filtering before retrieval, permissions in the tools, schema-validated outputs, injection tests that try to break it |
| Trade-offs | One approach | Notes what was considered and rejected, e.g. "chose X over Y because...; cost: ..." |
| Scale question | A generic "use a bigger DB" | Names the first bottleneck concretely: ingestion time, conflict detection that compares every pair of documents (O(n²)), token budget against more conflicting sources, re-indexing cost, plus a fix for each |
| Code | Works | Clean boundaries, typed, with ingestion separate from serving and the LLM easy to swap |
| Reliability | Happy path | Handles a timeout inside the 30 seconds, malformed LLM output, a retry or fallback, and partial ingestion failure |

## What to add to stand out
1. **An eval harness.** Run something like `make eval`, which prints a scorecard of route accuracy, citation precision and recall, and refusal correctness. Few candidates do this.
2. **Rule extraction at ingestion.** Precedence and conflict handling become deterministic and testable. This is the hardest part of the task, so solving it cleanly carries weight.
3. **A modified-corpus test.** Change a value in a document, re-ingest, and assert that the answer changes. This shows you thought about must-pass check 5.
4. **Adversarial tests.** Cover injection, role claims, bulk data requests, leaks about whether an employee exists, and session hijacking.
5. **A sharp DESIGN.md.** Two pages of decisions, trade-offs and honest failures, not a description of the code.
6. **Clean commit history.** Small, logical commits that tell the story of how you built it.

> Reviewers will judge seniority mostly from the eval harness, the rule-extraction design and an honest DESIGN.md. The required features only get you to the starting line.
