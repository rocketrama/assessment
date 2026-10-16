"""Section 5.1 precedence between eligible documents (TASK.pdf §5.1).

Tests in order, each only when the previous one does not decide:
authority (policy, addendum, memo, handbook, calendar outrank faq, wiki), then
scope (specific regions outrank GLOBAL), then recency (later effective_date).
Documents with equal keys are an unresolved conflict: both are kept.

`resolve` applies this to one rule across documents: the current winner(s), the
outranked facts (never answered from), and upcoming changes — a not-yet-effective
document is mentioned only if it would win on its own effective date (§5.1).
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from app.retrieval.corpus import FactRecord
from app.retrieval.eligibility import GLOBAL, DocMeta, can_read, covers_region

HIGH_AUTHORITY = frozenset({"policy", "addendum", "memo", "handbook", "calendar"})


def precedence_key(doc: DocMeta) -> tuple[int, int, str]:
    """Bigger wins. Compare whole tuples: authority, then scope, then recency."""
    authority = 1 if doc.doc_type in HIGH_AUTHORITY else 0
    scope = 0 if GLOBAL in doc.regions else 1
    return (authority, scope, doc.effective_date.isoformat())


def winners(docs: Sequence[DocMeta]) -> list[DocMeta]:
    """The top-precedence documents; more than one means an unresolved conflict."""
    if not docs:
        return []
    best = max(precedence_key(d) for d in docs)
    return sorted((d for d in docs if precedence_key(d) == best), key=lambda d: d.doc_id)


# --- resolving one rule across documents --------------------------------------

_NUMBER_WORDS = {
    w: str(i)
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
} | {"thirty": "30", "forty": "40", "fifty": "50", "sixty": "60"}
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def value_key(value: str) -> str:
    """Comparable form of a fact value: "ten working days" == "10" == "10 days";
    non-numeric values compare as normalised text."""
    words = [_NUMBER_WORDS.get(w, w) for w in re.findall(r"[a-z0-9.]+", value.lower())]
    numbers = _NUM_RE.findall(" ".join(words))
    return "#" + ",".join(str(float(n)) for n in numbers) if numbers else " ".join(words)


@dataclass(frozen=True)
class RuleResolution:
    rule_key: str
    winners: tuple[FactRecord, ...]  # facts of the top-precedence document(s)
    outranked: tuple[FactRecord, ...]  # eligible but losing facts: never answered from
    upcoming: tuple[FactRecord, ...]  # future facts that would win on their effective date

    @property
    def conflict(self) -> bool:
        """Unresolved: the winners disagree on the value (§5.1 rule 3)."""
        return len({value_key(f.value) for f in self.winners}) > 1

    @property
    def winning_doc_ids(self) -> tuple[str, ...]:
        return tuple(sorted({f.doc_id for f in self.winners}))


def resolve(
    rule_key: str,
    facts: Sequence[FactRecord],
    docs: Mapping[str, DocMeta],
    role: str,
    region: str,
    reference_date: date,
    include_upcoming: bool = True,
) -> RuleResolution | None:
    """Apply §5.1 to every fact of one rule. Unreadable or other-region documents
    are dropped first; None when nothing is current or upcoming.
    `include_upcoming=False` for questions about a past date: the spec's example
    answer for 2025-08-01 is the old value alone, with no change announced."""
    visible = [f for f in facts if can_read(role, docs[f.doc_id].access_level) and covers_region(docs[f.doc_id], region)]
    current = [f for f in visible if docs[f.doc_id].effective_date <= reference_date]
    future = [f for f in visible if docs[f.doc_id].effective_date > reference_date]

    win_ids = {d.doc_id for d in winners([docs[i] for i in sorted({f.doc_id for f in current})])}
    winning = tuple(f for f in current if f.doc_id in win_ids)
    outranked = tuple(f for f in current if f.doc_id not in win_ids)

    upcoming: list[FactRecord] = []
    for doc_id in sorted({f.doc_id for f in future} if include_upcoming else (), key=lambda i: (docs[i].effective_date, i)):
        on_date = docs[doc_id].effective_date
        contenders = [f for f in visible if docs[f.doc_id].effective_date <= on_date]
        top = {d.doc_id for d in winners([docs[i] for i in sorted({f.doc_id for f in contenders})])}
        if doc_id in top:
            upcoming.extend(f for f in future if f.doc_id == doc_id)

    if not winning and not upcoming:
        return None
    return RuleResolution(rule_key, winning, outranked, tuple(upcoming))
