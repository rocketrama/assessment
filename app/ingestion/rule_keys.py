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
Units are not part of the key (the LLM names them inconsistently); values never
are — the same rule with different values must meet so the conflict surfaces.
Region, date and document type are left out; precedence handles them.
Over-merging is worse than under-merging, so the threshold stays tight.

Input is sorted by fact_id, so the result is reproducible.
"""

import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import replace

import numpy as np

from app.ingestion.extract_facts import Fact

# Cosine distance. Measured on the assessment corpus (317 facts, bge-small):
# 0.12 merged only same-rule facts across documents; 0.15 added correct merges
# (v1/v2 bereavement, parental leave) but also one wrong one (sick-leave credit
# date with annual-leave accrual credit date); 0.20 merged clearly different
# rules. Over-merging is worse than under-merging, so the default is 0.12.
# RULE_CLUSTER_THRESHOLD overrides it.
DEFAULT_THRESHOLD = 0.12

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


def normalize_conditions(conditions: str | None) -> str:
    text = _norm_words(conditions)
    if text in _NONE_WORDS or _WHERE_ONLY_RE.match(text):
        return ""
    return text


def fact_key_text(fact: Fact) -> str:
    key = f"{fact.topic.strip().lower()}: {fact.attribute.strip().lower()}"
    conditions = normalize_conditions(fact.conditions)
    return f"{key} ({conditions})" if conditions else key


_KEY_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def key_numbers(key_text: str) -> frozenset[str]:
    return frozenset(_KEY_NUMBER_RE.findall(key_text))


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


def assign_rule_keys(
    facts: Sequence[Fact],
    embed: Callable[[list[str]], np.ndarray],
    threshold: float = DEFAULT_THRESHOLD,
) -> list[Fact]:
    ordered = sorted(facts, key=lambda f: f.fact_id)
    if not ordered:
        return []
    texts = sorted({fact_key_text(f) for f in ordered})
    vectors = dict(zip(texts, embed(texts)))

    keys = [fact_key_text(f) for f in ordered]
    matrix = np.stack([vectors[k] for k in keys])
    docs = np.array([f.doc_id for f in ordered])
    key_arr = np.array(keys)
    numbers = [key_numbers(k) for k in keys]
    same_doc_other_key = (docs[:, None] == docs[None, :]) & (key_arr[:, None] != key_arr[None, :])
    different_numbers = np.array([[bool(a and b and a != b) for b in numbers] for a in numbers])
    labels = average_linkage_labels(matrix, threshold, same_doc_other_key | different_numbers)

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
