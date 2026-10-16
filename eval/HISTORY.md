# Eval history

Same 48 cases × 3 runs (`eval/cases.yaml`), real service, model `nvidia:openai/gpt-oss-20b`
(`LLM_REASONING_EFFORT=low`), dev DB, APP_TODAY 2026-09-15. The latest full report is
`eval/report.md`; this file keeps the scorecard of every round and what changed between them.

| Metric | Baseline | Round 1 | Round 2 |
|---|---|---|---|
| Cases fully correct (case × run) | 110/150 (73%) | 131/150 (87%) | 137/150 (91%) |
| Route accuracy | 79% | 91% | 96% |
| Required citations present (recall) | 79% | 100% | 100% |
| Citation precision | 92% | 97% | 99% |
| Outdated / restricted documents not cited | 92% | 97% | 100% |
| Key facts in the answer | 71% | 89% | 91% |
| Wrong or forbidden content avoided | 100% | 100% | 94% |
| Tool calls as expected | 53% | 70% | 87% |
| Consistent across 3 runs | 74% | 75% | 72% |
| Latency median / p95 / max (limit 30 s) | 3.2 / 6.1 / 10.1 s | 3.7 / 8.1 / 12.2 s | 3.6 / 7.4 / 17.8 s |

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
