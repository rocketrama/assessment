"""Section 5.1 precedence between eligible documents (TASK.pdf §5.1).

Tests in order, each only when the previous one does not decide:
authority (policy, addendum, memo, handbook, calendar outrank faq, wiki), then
scope (specific regions outrank GLOBAL), then recency (later effective_date).
Documents with equal keys are an unresolved conflict: both are kept.
"""

from collections.abc import Sequence

from app.retrieval.eligibility import GLOBAL, DocMeta

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
