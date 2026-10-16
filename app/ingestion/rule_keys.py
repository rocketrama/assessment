"""Group facts that state "the same rule" under one rule_key (ARCHITECTURE §5.5).

Facts are first partitioned by normalised (unit, conditions) — hard constraints,
so e.g. "levels L1-L3" and "levels L4+" can never merge. Within a partition the
fact keys ("topic: attribute") are embedded locally and clustered with average
linkage on cosine distance at a conservative threshold (no single linkage: it
chains unrelated rules). Implemented in numpy rather than scipy: it is a few
lines, and one dependency fewer to break. Values are never part
of the key: the same rule with different values must land together so the
conflict surfaces. A document states each rule once, so two facts from the same
document with different keys are never merged (cannot-link): this keeps e.g.
"primary caregiver weeks" and "secondary caregiver weeks" apart even though
their keys embed almost identically. Region, date and document type are left out too; precedence
handles them. Over-merging is worse than under-merging (a wrong silent winner
vs a missed conflict), so the threshold stays tight.

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


def normalize_unit(unit: str | None) -> str:
    """'Working days' / 'days' / 'business day' → 'day'; other units lowercased and singular."""
    text = _norm_words(unit)
    if text in _NONE_WORDS:
        return ""
    text = re.sub(r"\b(working|business)\s+", "", text)
    return " ".join(w[:-1] if w.endswith("s") and len(w) > 3 else w for w in text.split())


def normalize_conditions(conditions: str | None) -> str:
    text = _norm_words(conditions)
    return "" if text in _NONE_WORDS else text


def fact_key_text(fact: Fact) -> str:
    return f"{fact.topic.strip().lower()}: {fact.attribute.strip().lower()}"


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
    (Partitions hold at most a few hundred keys, so the O(n³) loop is cheap.)"""
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
    partitions: dict[tuple[str, str], list[Fact]] = defaultdict(list)
    for fact in ordered:
        partitions[(normalize_unit(fact.unit), normalize_conditions(fact.conditions))].append(fact)

    texts = sorted({fact_key_text(f) for f in ordered})
    vectors = dict(zip(texts, embed(texts))) if texts else {}

    keyed: dict[str, Fact] = {}
    used: set[str] = set()
    for (unit, conditions), members in sorted(partitions.items()):
        matrix = np.stack([vectors[fact_key_text(f)] for f in members])
        docs = np.array([f.doc_id for f in members])
        key_texts = np.array([fact_key_text(f) for f in members])
        cannot_link = (docs[:, None] == docs[None, :]) & (key_texts[:, None] != key_texts[None, :])
        labels = average_linkage_labels(matrix, threshold, cannot_link)
        clusters: dict[int, list[Fact]] = defaultdict(list)
        for label, fact in zip(labels, members):
            clusters[int(label)].append(fact)
        for group in sorted(clusters.values(), key=lambda g: g[0].fact_id):
            base = _slug(fact_key_text(group[0]))
            suffix = "|".join(p for p in (_slug(unit), _slug(conditions)) if p)
            key = f"{base}|{suffix}" if suffix else base
            unique, n = key, 2
            while unique in used:
                unique, n = f"{key}#{n}", n + 1
            used.add(unique)
            for fact in group:
                keyed[fact.fact_id] = replace(fact, rule_key=unique)
    return [keyed[f.fact_id] for f in facts]
