"""Section 5.1 eligibility (TASK.pdf §3.1, §5.1). Pure functions over document metadata.

A document can be used only if the requester's role can read its access_level,
its regions include the requester's region or GLOBAL, and its effective_date is
on or before the reference date.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

GLOBAL = "GLOBAL"

# §3.1: which access levels each role can read.
READABLE_ACCESS = {
    "employee": frozenset({"all"}),
    "manager": frozenset({"all", "managers"}),
    "hr": frozenset({"all", "managers", "hr"}),
}


@dataclass(frozen=True)
class DocMeta:
    doc_id: str
    doc_type: str
    regions: tuple[str, ...]
    effective_date: date
    access_level: str


def can_read(role: str, access_level: str) -> bool:
    return access_level in READABLE_ACCESS.get(role, frozenset())


def covers_region(doc: DocMeta, region: str) -> bool:
    return GLOBAL in doc.regions or region in doc.regions


def is_eligible(doc: DocMeta, role: str, region: str, reference_date: date) -> bool:
    return can_read(role, doc.access_level) and covers_region(doc, region) and doc.effective_date <= reference_date


def is_upcoming(doc: DocMeta, role: str, region: str, reference_date: date) -> bool:
    """Readable and for the region, but not yet effective: a candidate to be
    mentioned as an upcoming change, never used as the current answer."""
    return can_read(role, doc.access_level) and covers_region(doc, region) and doc.effective_date > reference_date


def allowed_chunk_ids(
    chunk_docs: Mapping[str, str], docs: Mapping[str, DocMeta], role: str, region: str, reference_date: date
) -> frozenset[str]:
    """Chunks of eligible documents: the only chunks retrieval may search (§7.2)."""
    ok = {d.doc_id for d in docs.values() if is_eligible(d, role, region, reference_date)}
    return frozenset(c for c, doc_id in chunk_docs.items() if doc_id in ok)


def upcoming_chunk_ids(
    chunk_docs: Mapping[str, str], docs: Mapping[str, DocMeta], role: str, region: str, reference_date: date
) -> frozenset[str]:
    ok = {d.doc_id for d in docs.values() if is_upcoming(d, role, region, reference_date)}
    return frozenset(c for c, doc_id in chunk_docs.items() if doc_id in ok)
