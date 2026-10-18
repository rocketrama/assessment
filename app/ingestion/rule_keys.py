"""Group facts that state "the same rule" under one rule_key (ARCHITECTURE §5.5).

Each fact's key is "topic: attribute (conditions)" — conditions are part of the
compared text, not a separate exact-match partition. The LLM is inconsistent
about where it writes a qualifier: the accrual schedule put the service tier
in conditions ("less than 2 years") while v3 wrote it into the topic ("accrual:
less than 2 years of service"); an exact partition kept them apart, so the rule
table showed no current accrual rate at all (eval finding).

Keys are embedded locally and clustered with average linkage on cosine distance
at a conservative threshold (no single linkage: it chains unrelated rules), in
numpy rather than scipy (blocked on the dev machine; a few lines). Two hard
cannot-link constraints, deterministic and untuned:
- same document, different key: a document states each rule once ("primary
  caregiver weeks" vs "secondary caregiver weeks" embed almost identically);
- different numbers in the keys: tiers and levels ("less than 2 years" vs
  "2 to less than 5 years", "L1-L3" vs "L4-L5") are only ~0.09 apart as
  embeddings, but their numbers differ, so they are different rules.
- recommendation vs rule: "annual leave: encouraged minimum days" and
  "annual leave: days per year" embed closely, but a recommendation is not an
  entitlement; merged, the recommended ten days outranked Germany's 30 and was
  answered as India's entitlement (eval finding).
Validity wording in the attribute ("expiry", "lapse", "valid until") is normalised to
"deadline to use" before embedding, so a rule stated in the negative meets the rule it
denies (`normalize_attribute`).
Units are not part of the key (the LLM names them inconsistently); values never
are — the same rule with different values must meet so the conflict surfaces.
Region, date and document type are left out; precedence handles them.
Over-merging is worse than under-merging, so the threshold stays tight.

Near misses are judged by the LLM (`judge`, see rule_judge.py). An embedder can't
tell that "hybrid work", "home office" and "remote work" name one rule while
"primary caregiver weeks" and "secondary caregiver weeks" don't: the WFH memo sat
0.136 from the policy it replaces, the caregiver pair closer still. So after
clustering, cluster pairs whose average distance is above the threshold but within
NEAR_MISS_MAX — and that no cannot-link forbids — are asked "same rule?" and merged
on yes, closest first; a merge that would join a cannot-link pair is skipped. No
answer (no LLM, deadline, failed call) means no merge, the conservative default.

Input is sorted by fact_id, so the result is reproducible.
"""

import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import replace
from itertools import combinations

import numpy as np

from app.ingestion.extract_facts import Fact

# Cosine distance. Measured on the assessment corpus (317 facts, bge-small):
# 0.12 merged only same-rule facts across documents; 0.15 added correct merges
# (v1/v2 bereavement, parental leave) but also one wrong one (sick-leave credit
# date with annual-leave accrual credit date); 0.20 merged clearly different
# rules. Over-merging is worse than under-merging, so the default is 0.12.
# RULE_CLUSTER_THRESHOLD overrides it.
DEFAULT_THRESHOLD = 0.12
# Upper end of the near-miss band sent to the LLM judge. Measured on the assessment
# corpus (303 facts, 215 clusters): 149 cluster pairs up to 0.22, 390 up to 0.25, and
# the pairs between 0.22 and 0.25 were all different rules. The WFH memo joins the
# policy at 0.136 and the FAQ/wiki cluster joins them at 0.170.
NEAR_MISS_MAX = 0.22
# Facts shown per cluster in a judge question: enough to see the rule, few tokens.
FACTS_PER_SIDE = 3

# (cluster A facts, cluster B facts) pairs → one verdict each: True = same rule,
# False = different, None = no answer (treated as different).
Judge = Callable[[list[tuple[list["Fact"], list["Fact"]]]], list[bool | None]]

_NONE_WORDS = {"", "none", "null", "n/a", "na", "all", "all employees", "any", "-"}


def _norm_words(text: str | None) -> str:
    raw = (text or "").strip().lower()
    if raw in _NONE_WORDS:
        return ""
    return " ".join(re.sub(r"[^a-z0-9.+]+", " ", raw.replace("–", "-")).split())


# "employees in india", "staff based in the berlin office", "for employees of veltrix gmbh":
# says only who works where. The document's region comes from the manifest, so this is
# no subgroup at all — the prompt forbids it, but the live model still writes it.
_CONDITION_WORDS = r"after|before|with|without|who|whose|during|while|on|from|until|if|when|over|under|than|and|or|for"
_PLACE = rf"(?:(?!(?:{_CONDITION_WORDS})\b)[a-z.+-]+ ?){{1,4}}"  # at most 4 words, none starting a real condition
_WHERE_ONLY_RE = re.compile(
    rf"^(?:for |applies to )?(?:all )?(?:employees|employee|staff)"
    rf"(?: (?:in|based in|of|at|working in|employed by|located in)(?: the)? {_PLACE})?(?: office)?$"
)


# "bengaluru and austin offices", "the london office": names offices, so where, not who
# (eval finding: the WFH memo's rule stayed apart from the policy it replaces). One place
# alone without "office" ("bengaluru, india") is kept: a table with a row per office
# in one document states a different value per row. Home / remote offices are real conditions.
_OFFICES_RE = re.compile(
    rf"^(?:for |applies to )?(?:(?:all )?(?:employees|employee|staff) )?(?:(?:in|at|of|based in|working in|located in) )?"
    rf"(?:the )?(?!(?:home|remote|virtual)\b){_PLACE}(?:(?:and|or) (?:the )?{_PLACE})*(?:offices?|locations?|sites?)$"
)


def normalize_conditions(conditions: str | None) -> str:
    text = _norm_words(conditions)
    if text in _NONE_WORDS or _WHERE_ONLY_RE.match(text) or _OFFICES_RE.match(text):
        return ""
    return text


# Validity wording → one attribute, before embedding. The extraction prompt asks for a
# negative rule to reuse the positive rule's attribute, but the live model doesn't always:
# a cold Docker rebuild wrote the India addendum's "do not expire on a fixed date" as
# "expiry" and the global "31 March" as "deadline to use". As embeddings they sat 0.113
# apart (threshold 0.12) and a closer wrong match won, so precedence never compared them
# and India was told "31 March" (B1 finding). Normalised, the two are 0.023 apart.
# General vocabulary, not corpus facts. Forfeiture of unused days stays a separate rule.
_VALIDITY_RE = re.compile(
    r"\b(?:expiry|expiration|expires?|lapse|lapses|validity|valid until|use[- ]by|must be used by)"
    r"(?: (?:date|period|deadline))?\b"
)


def normalize_attribute(attribute: str) -> str:
    return " ".join(_VALIDITY_RE.sub("deadline to use", attribute.strip().lower()).split())


def fact_key_text(fact: Fact) -> str:
    key = f"{fact.topic.strip().lower()}: {normalize_attribute(fact.attribute)}"
    conditions = normalize_conditions(fact.conditions)
    return f"{key} ({conditions})" if conditions else key


_KEY_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_RECOMMENDATION_RE = re.compile(r"\b(?:encouraged|recommended|suggested|advised|should)\b")


def key_numbers(key_text: str) -> frozenset[str]:
    return frozenset(_KEY_NUMBER_RE.findall(key_text))


def is_recommendation(key_text: str) -> bool:
    return bool(_RECOMMENDATION_RE.search(key_text))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def average_linkage_labels(
    vectors: np.ndarray, threshold: float, cannot_link: np.ndarray | None = None
) -> list[int]:
    """Agglomerative clustering with average linkage on cosine distance, merging
    while the closest pair of clusters is within `threshold`. Rows must be unit
    length. `cannot_link[i, j]` True forbids rows i and j from ever sharing a
    cluster (their distance is infinite, and the average stays infinite after
    merges). Ties merge the lowest indices first, so the result is deterministic.
    (A corpus has a few hundred facts, so the O(n³) loop is cheap.)"""
    n = len(vectors)
    labels = list(range(n))
    if n < 2:
        return labels
    dist = 1.0 - np.clip(vectors @ vectors.T, -1.0, 1.0)
    if cannot_link is not None:
        dist[cannot_link] = np.inf
    np.fill_diagonal(dist, np.inf)
    sizes = np.ones(n)
    active = np.ones(n, dtype=bool)
    members = {i: [i] for i in range(n)}
    while active.sum() > 1:
        masked = np.where(np.outer(active, active), dist, np.inf)
        i, j = divmod(int(np.argmin(masked)), n)
        if masked[i, j] > threshold:
            break
        i, j = min(i, j), max(i, j)
        # Lance–Williams update for average linkage: size-weighted mean distance.
        merged = (sizes[i] * dist[i] + sizes[j] * dist[j]) / (sizes[i] + sizes[j])
        dist[i, :] = dist[:, i] = merged
        dist[i, i] = np.inf
        sizes[i] += sizes[j]
        active[j] = False
        members[i].extend(members.pop(j))
    for cluster, rows in members.items():
        for row in rows:
            labels[row] = cluster
    return labels


def cannot_link_matrix(facts: Sequence[Fact], keys: Sequence[str]) -> np.ndarray:
    """True where two facts may never share a rule key (see module docstring)."""
    docs = np.array([f.doc_id for f in facts])
    key_arr = np.array(keys)
    numbers = [key_numbers(k) for k in keys]
    advisory = np.array([is_recommendation(k) for k in keys])
    same_doc_other_key = (docs[:, None] == docs[None, :]) & (key_arr[:, None] != key_arr[None, :])
    different_numbers = np.array([[bool(a and b and a != b) for b in numbers] for a in numbers])
    different_strength = advisory[:, None] != advisory[None, :]
    return same_doc_other_key | different_numbers | different_strength


def near_miss_pairs(
    labels: Sequence[int], vectors: np.ndarray, cannot_link: np.ndarray, low: float, high: float
) -> list[tuple[float, int, int]]:
    """(average distance, cluster a, cluster b) for cluster pairs with low < distance <= high
    and no cannot-link between them, closest first (ties by label: deterministic).
    Pairs kept apart by the same-document constraint are not asked: measured on the
    assessment corpus, 67 such pairs within the threshold were nearly all different rules
    (Berlin vs Munich allowance, band minimum vs maximum, primary vs secondary caregiver)."""
    members: dict[int, list[int]] = defaultdict(list)
    for row, label in enumerate(labels):
        members[label].append(row)
    dist = 1.0 - np.clip(vectors @ vectors.T, -1.0, 1.0)
    pairs = []
    for a, b in combinations(sorted(members), 2):
        block = np.ix_(members[a], members[b])
        if cannot_link[block].any():
            continue
        d = float(dist[block].mean())
        if low < d <= high:
            pairs.append((d, a, b))
    return sorted(pairs)


def _representatives(rows: list[int], facts: Sequence[Fact], keys: Sequence[str]) -> list[Fact]:
    """One fact per distinct key text, in fact_id order, at most FACTS_PER_SIDE."""
    seen: set[str] = set()
    out: list[Fact] = []
    for row in rows:
        if keys[row] not in seen:
            seen.add(keys[row])
            out.append(facts[row])
    return out[:FACTS_PER_SIDE]


def merge_judged(
    labels: list[int],
    vectors: np.ndarray,
    cannot_link: np.ndarray,
    facts: Sequence[Fact],
    keys: Sequence[str],
    judge: Judge,
    low: float,
    high: float = NEAR_MISS_MAX,
) -> list[int]:
    """Labels after merging the near-miss cluster pairs the judge says are one rule."""
    pairs = near_miss_pairs(labels, vectors, cannot_link, low, high)
    if not pairs:
        return labels
    members: dict[int, list[int]] = defaultdict(list)
    for row, label in enumerate(labels):
        members[label].append(row)
    questions = [
        (_representatives(members[a], facts, keys), _representatives(members[b], facts, keys)) for _, a, b in pairs
    ]
    verdicts = judge(questions)

    root = {label: label for label in members}

    def find(label: int) -> int:
        while root[label] != label:
            label = root[label]
        return label

    groups = {label: list(rows) for label, rows in members.items()}
    for (_, a, b), same in zip(pairs, verdicts):
        ra, rb = find(a), find(b)
        if not same or ra == rb or cannot_link[np.ix_(groups[ra], groups[rb])].any():
            continue
        keep, drop = min(ra, rb), max(ra, rb)
        root[drop] = keep
        groups[keep].extend(groups.pop(drop))
    return [find(label) for label in labels]


def assign_rule_keys(
    facts: Sequence[Fact],
    embed: Callable[[list[str]], np.ndarray],
    threshold: float = DEFAULT_THRESHOLD,
    judge: Judge | None = None,
) -> list[Fact]:
    ordered = sorted(facts, key=lambda f: f.fact_id)
    if not ordered:
        return []
    texts = sorted({fact_key_text(f) for f in ordered})
    vectors = dict(zip(texts, embed(texts)))

    keys = [fact_key_text(f) for f in ordered]
    matrix = np.stack([vectors[k] for k in keys])
    forbidden = cannot_link_matrix(ordered, keys)
    labels = average_linkage_labels(matrix, threshold, forbidden)
    if judge is not None:
        labels = merge_judged(labels, matrix, forbidden, ordered, keys, judge, low=threshold)

    clusters: dict[int, list[Fact]] = defaultdict(list)
    for label, fact in zip(labels, ordered):
        clusters[int(label)].append(fact)
    keyed: dict[str, Fact] = {}
    used: set[str] = set()
    for group in sorted(clusters.values(), key=lambda g: g[0].fact_id):
        key = _slug(fact_key_text(group[0]))
        unique, n = key, 2
        while unique in used:
            unique, n = f"{key}#{n}", n + 1
        used.add(unique)
        for fact in group:
            keyed[fact.fact_id] = replace(fact, rule_key=unique)
    return [keyed[f.fact_id] for f in facts]
