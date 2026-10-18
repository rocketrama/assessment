# Eval history

50 cases × 3 runs up to round 3, 53 from round 4 (3 reworded questions added), so round 4
percentages are over a slightly harder set (`eval/cases.yaml`), real service, model `nvidia:openai/gpt-oss-20b`
(`LLM_REASONING_EFFORT=low`), dev DB, APP_TODAY 2026-09-15. The latest full report is
`eval/report.md`; this file keeps the scorecard of every round and what changed between them.

| Metric | Baseline | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 | Round 6 |
|---|---|---|---|---|---|---|---|
| Cases fully correct (case × run) | 110/150 (73%) | 131/150 (87%) | 137/150 (91%) | 142/150 (95%) | 148/159 (93%) | 142/159 (89%) | 148/159 (93%) |
| Route accuracy | 79% | 91% | 96% | 99% | 94% | 92% | 96% |
| Required citations present (recall) | 79% | 100% | 100% | 98% | 95% | 88% | 93% |
| Citation precision | 92% | 97% | 99% | 98% | 92% | 94% | 98% |
| Outdated / restricted documents not cited | 92% | 97% | 100% | 97% | 100% | 99% | 100% |
| Key facts in the answer | 71% | 89% | 91% | 94% | 91% | 86% | 91% |
| Wrong or forbidden content avoided | 100% | 100% | 94% | 100% | 100% | 100% | 100% |
| Tool calls as expected | 53% | 70% | 87% | 97% | 90% | 87% | 100% |
| Consistent across 3 runs | 74% | 75% | 72% | 79% | 70% | 80% | 75% |
| Latency median / p95 / max (limit 30 s) | 3.2 / 6.1 / 10.1 s | 3.7 / 8.1 / 12.2 s | 3.6 / 7.4 / 17.8 s | 3.9 / 9.3 / 21.0 s | 3.7 / 9.3 / 26.0 s | 4.2 / 19.3 / 26.0 s | 3.2 / 16.4 / **35.1** s |

## Baseline → round 1

- Facts: rules stated in the negative ("carried forward days do not expire") are extracted.
- Citations: a passage is dropped only when it holds an outranked fact and none of its winning
  values is in the answer (the India addendum was being dropped).
- Dates: "from 21 to 23 September" (bare start day) and a weekday before a full date.
- Planner guidance: rule questions are policy questions; HR topics are never off-topic;
  entitlement ≠ balance; future balances need documents and tools.
- Composer: requester level and join date in context (not personal data, §7.4); a rule only in a
  future document is answered as "not in effect yet"; prefer higher-authority passages.
- Deterministic bulk check ("every employee", "all staff", …).
- Region-only conditions ("employees in India") count as no condition in rule grouping.

## Round 1 → round 2

- Rule grouping: conditions are part of the compared key text instead of an exact-match partition
  (the accrual schedule wrote the service tier as a condition, v3 into the topic, so the rule table
  showed no current accrual rate); a numeric cannot-link keeps tiers and levels apart.
- Extraction prompt: qualifiers go in conditions.
- The user's own words beat the planner: leave type and date ranges are read from the message;
  team-leave ranges ("between 5 and 9 October") too.
- Bulk check also when the planner calls a bulk request a policy question.
- Eval: `no_answer` reasons recorded; an FAQ that agrees with the policy may be cited.

## Found in round 2, fixed for round 3

- **Regression (hallucination):** "Can I take half a day?" was answered "yes" with a made-up policy
  quote. The only passage with the answer was the FAQ ("No. All leave is taken in whole days"), and
  the round-1 instruction "rely on the higher-authority passage" made the model ignore it. The
  instruction now says an FAQ/wiki is a valid source when no higher-authority passage covers the
  point, and never to state what no passage says.
- Cancel "on 6 October": the date is read from the message when the planner drops it.
- A spurious "missing: leave type" from the planner no longer blocks a balance lookup.
- Future balance: the composer prompt describes the calculation steps generically (rate for the
  employee's length of service, credits until the target date, limits).
- New over-merges from the round-2 grouping are listed in DESIGN.md §6 (e.g. secondary-caregiver
  "4 weeks" with "two blocks" and "within six months").
- Remaining run-to-run variation is mostly provider timeouts on the free NVIDIA endpoint
  (requests that end in the safe "please try again") and planner variance.

## Round 3: what is still wrong (not fixed)

- **Future balance** ("how many days will I have by December?"): 0/3. The model doesn't do the
  accrual arithmetic reliably (answers 12, 33 and 10 days; or claims there is no accrual rate).
  Prompt guidance didn't fix it. The real fix is the one in DESIGN §3: compute the arithmetic in
  code from the tool balance and the accrual facts, and let the model only explain it.
- **Half days**: the invented "yes, half days are allowed" still appears in 1 of 3 runs (was 3/3).
  The FAQ with the right answer is retrieved; the model sometimes ignores it. The passage that
  says it in the policy (v2 §11 "whole days") isn't in the top 10 hits for that wording.
- **Team leave**: sometimes lists only the approved request and omits the pending one.
- **WFH for India**: 1 of 3 runs answered 2 days instead of the memo's 3.
- One request ended in the safe "please try again" (provider timeout); slowest reply 21.0 s.
- These are answer-writing failures of a small model at low reasoning effort; the deterministic
  parts (access, precedence table, routes, tools, dates) were correct in every run.

## Round 3 → round 4

- Search: BM25 counts half in the rank fusion (vectors 1.0, BM25 0.5). At equal weight, BM25's
  matches on common words pushed the bereavement passages to #11–12; they are now in the top 10.
  Three reworded questions added to the eval (sick leave, bereavement, work from home).
- Extraction: FAQ yes/no answers are extracted as facts; a chunk whose batch fails is retried alone
  (extraction version 9, cache rebuilt: 30 documents, 145 chunks, no failed chunks, 188 s cold).
- Tool-only answers (balances, requests, team leave) are written by code from the records, with no
  LLM call (`app/orchestration/tool_answers.py`).
- Future balances: code counts the accrual credit dates up to the target date
  (`app/orchestration/future_balance.py`); the model applies the rate from the documents.
- Office names ("Bengaluru and Austin offices") count as a location condition, so the India WFH memo
  groups with the remote-work policy.
- FAQ/wiki-only rules are marked in the rule table, and their citations are dropped when a policy
  covers the topic.

Round 3 failures now fixed: half days 3/3 (was 2/3), team leave 3/3, WFH India 3/3, bereavement 3/3
with no FAQ citation. Future balance improved to 1/3 (was 0/3).

## Round 4: what is still wrong (not fixed)

- **Unusable planner output (3 of 11 failures)**: the planner returned invalid JSON on the first try
  and on the repair, so the turn ended in "please try again" (`unknown-employee-balance`,
  `manager-report-balance`, `role-claim`). Each passed in the other 2 runs. This wasn't logged; the
  planner now logs `planner failed` with `error=unusable plan`.
- **Provider timeout (1 failure)**: composer timed out after 12.9 s and the retry didn't fit the
  request deadline (`apply-modify` turn 2). Slowest reply was 26.0 s, close to the 30 s limit.
- **Future balance**: 1/3. In one run the planner asked for the balance only, with no document query,
  so the answer said the policy has no credit date. The accrual table does have it.
- **Retrieval or answer misses (1 run each)**: `sick-days-india` and `pip-length-hr` ended in
  `no_answer` (no supported citation); `carry-forward-us` cited only the global policy (10 days)
  instead of the US addendum (5); `paraphrase-wfh` ("Can I do my job from my house?") was refused
  as off-topic.
- **Scorer false negative**: `carry-deadline-india` run 1 answered correctly ("does not have a fixed
  expiration date"), but the expected phrases don't include that wording.
- **Extraction gap**: the FAQ's "No" on half days (FAQ-LEAVE::008) still yields no fact; the policy's
  "whole days" rule (v2 §11) answers the question.
- Consistency dropped to 70% (from 79%). Most of the inconsistent cases have one run hit by the
  planner or timeout failures above.

## Round 4 → round 5

- Search: dense retrieval moved from a numpy dot product to FAISS `IndexFlatIP` with an
  `IDSelectorBatch` of the allowed chunks. On all 56 eval questions its top 10 is identical to
  numpy's (max score difference 1.2e-7), so it is not a factor in the results below.
- Extraction version 10 (made in another session after a Postman finding): the prompt keeps a rule's
  strength ("encouraged to take at least ten days" is a recommendation, not "minimum days per year"),
  and rule grouping never merges a recommendation with an entitlement. Facts were rebuilt with it.
- The planner logs `planner failed error=unusable plan`.
- Provider: same NVIDIA model, but much slower today: 21 planner and 8 composer calls timed out on
  the first attempt (p95 latency 19.3 s, was 9.3 s).

## Round 5: what is wrong

- **Regression, wrong answer (carry-deadline-india 0/3):** "use carried-forward leave by 31 March"
  for India. With version-10 facts the India addendum's rule is `carry-forward-expiration` ("do not
  expire on a fixed date") and the global one is `carry-forward-deadline-to-use` ("31 March"), so
  precedence never compares them and the global value is the only one in its rule. In version 9 they
  were one rule. Same cause: policy v1's "up to 3" bereavement days and v1/v2 "whole days" are now
  separate rules, and `paraphrase-sick` cited the outdated v1 once.
- **Provider timeouts (3 failures):** "please try again" on `holiday-thanksgiving-friday`,
  `own-sick-balance`, `unknown-employee-balance`.
- **`no_answer` with "no supported citation":** `pip-length-hr` 0/3 (was 2/3), `sick-days-india`
  and `holiday-restricted` 1 run each. Not yet diagnosed.
- **Future balance 0/3:** the calendar note lists month ends (4) and month starts (3) separately; the
  model added both (7 credits, 22.5 days instead of 18).
- Consistency rose to 80% (from 70%): fewer answers differed between runs, even though more cases
  failed in every run.

## Round 5 → round 6

- Extraction version 11: a rule stated in the negative uses the attribute of the positive rule it
  denies (prompt examples are not from the corpus). Rebuild: 295 facts; the India addendum's "do not
  expire on a fixed date" is again in the same rule as the global "31 March", and policy v1's
  bereavement "3" and v1/v2 "whole days" group with v2. Six code-of-conduct chunks timed out and got
  no facts (they stay retrievable; retried on the next start).
- Future balance: `future_balance.credit_day` reads the credit day from the requester's eligible
  facts ("last calendar day of each month" → month ends) and the calendar note lists only those
  dates. With no clear credit-day fact the note still lists both.

## Round 6: what is still wrong

- **Over the 30 s limit:** `role-claim` run 1 took 35.1 s (a planner call, then a planner repair that
  timed out at 12 s). The request deadline only stops new calls from starting; a call already
  running is not cut off at the deadline.
- **`no_answer` with "no supported citation" (5 failures):** `sick-days-india`, `sick-days-us`,
  `pip-length-hr` (one run each) and `carry-deadline-india` (two runs) ended in "the documents
  don't cover this" although the right passages exist. Now the largest failure group; not yet
  diagnosed.
- **Future balance 0/3:** the answer said the documents give no accrual rate or credit date, so the
  accrual passages were not in the context for that wording; the credit-day fix had nothing to read.
- `wfh-india-memo` answered 2 days once (memo not cited); `role-claim` once ended in "please try
  again" (planner timeouts: 9 planner, 4 composer timeouts in the run).
- `carry-deadline-india` passed in 1 of 3 runs (was 0/3); the two failures were "no supported
  citation", not the wrong "31 March".
