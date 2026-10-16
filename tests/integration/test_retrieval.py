"""Retrieval end to end: ingestion (fake LLM) → corpus → search → precedence → context.

The corpus mirrors the spec's scenario: v1 / v2 / outdated FAQ / regional
addendum / future v3, plus manager-only and HR-only documents.
"""

import json
import re
from datetime import date

import pytest

from app.config import Settings
from app.data.employees import get_employee
from app.ingestion.cache import cache_db_path
from app.ingestion.loaders.ocr import PageOcr
from app.ingestion.pipeline import run_ingestion
from app.llm.client import CORPUS_TOKEN_LIMIT, LLMClient
from app.retrieval.retriever import Retriever
from tests.conftest import HashingEmbedder

TODAY = date(2026, 9, 15)

# doc_id: (doc_type, regions, effective_date, access_level, sentence)
DOCS = {
    "POL-V1": ("policy", ["GLOBAL"], "2025-01-01", "all", "Staff get 10 days of sick leave each year."),
    "POL-V2": ("policy", ["GLOBAL"], "2026-04-01", "all", "Staff get 12 days of sick leave each year."),
    "FAQ": ("faq", ["GLOBAL"], "2026-05-01", "all", "Staff get 10 days of sick leave each year."),
    "ADD-DE": ("addendum", ["DE"], "2025-06-01", "all", "Staff get 30 days of sick leave each year."),
    "POL-V3": ("policy", ["GLOBAL"], "2027-01-01", "all", "Staff get 14 days of sick leave each year."),
    "HB-MGR": ("handbook", ["GLOBAL"], "2026-01-01", "managers", "Confidential approval matrix: managers grant 7 days of sick leave extensions."),
    "HR-BANDS": ("policy", ["GLOBAL"], "2026-01-01", "hr", "Confidential salary bands: band L3 pays 9 days of sick leave top-up."),
}


class RegexFactProvider:
    """A fake extraction model: one fact per '<N> days of <x> leave' phrase."""

    def chat(self, model, system, user, max_tokens, timeout):
        facts = []
        for chunk_id, body in re.findall(r'<document id="([^"]+)">\n(.*?)\n</document>', user, re.DOTALL):
            for m in re.finditer(r"(\d+) days of (\w+ leave)", body):
                facts.append({"chunk_id": chunk_id, "topic": m.group(2), "attribute": "days per year",
                              "value": m.group(1), "unit": "days", "conditions": None, "quote": m.group(0)})
        return json.dumps({"facts": facts})


@pytest.fixture(scope="module")
def retriever(tmp_path_factory):
    root = tmp_path_factory.mktemp("retrieval")
    corpus = root / "corpus"
    corpus.mkdir()
    manifest = []
    for doc_id, (doc_type, regions, effective, access, sentence) in DOCS.items():
        (corpus / f"{doc_id}.md").write_text(f"# {doc_id}\n\n## Sick leave\n\n{sentence}\n", encoding="utf-8")
        manifest.append({"doc_id": doc_id, "file": f"{doc_id}.md", "title": doc_id, "doc_type": doc_type,
                         "regions": regions, "effective_date": effective, "access_level": access})
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    settings = Settings(_env_file=None, corpus_dir=str(corpus), manifest_path=str(root / "manifest.json"),
                        cache_dir=str(root / "cache"))
    report = run_ingestion(settings, PageOcr(vision=None, use_tesseract=False),
                           llm=LLMClient(RegexFactProvider(), "fake"), embedder=HashingEmbedder())
    assert report.facts.kept == len(DOCS) and report.rule_keys == 1
    return Retriever.load(cache_db_path(settings.cache_dir), HashingEmbedder())


def ask(retriever, db_conn, employee_id, query="How many sick leave days do I get?", reference=TODAY):
    return retriever.retrieve(query, get_employee(db_conn, employee_id), reference, today=TODAY)


def docs_in_context(result):
    return sorted({cid.split("::")[0] for cid in result.context.chunk_ids})


def test_india_employee_gets_v2_with_v3_as_upcoming(retriever, db_conn):
    result = ask(retriever, db_conn, "E1001")
    (rule,) = result.resolutions
    assert [(f.doc_id, f.value) for f in rule.winners] == [("POL-V2", "12")]
    assert sorted(f.doc_id for f in rule.outranked) == ["FAQ", "POL-V1"]
    assert [(f.doc_id, f.value) for f in rule.upcoming] == [("POL-V3", "14")]
    # Outdated sources whose every fact lost are not in the context, so they can't be cited.
    assert docs_in_context(result) == ["POL-V2", "POL-V3"]


def test_germany_employee_gets_addendum_and_no_v3(retriever, db_conn):
    result = ask(retriever, db_conn, "E1003")
    (rule,) = result.resolutions
    assert rule.winning_doc_ids == ("ADD-DE",) and rule.upcoming == ()
    assert docs_in_context(result) == ["ADD-DE"]


def test_past_date_uses_v1_and_announces_nothing(retriever, db_conn):
    result = ask(retriever, db_conn, "E1001", query="How many sick leave days were there?", reference=date(2025, 8, 1))
    (rule,) = result.resolutions
    assert rule.winning_doc_ids == ("POL-V1",) and rule.upcoming == ()
    assert docs_in_context(result) == ["POL-V1"]


RESTRICTED = {"HB-MGR": {"manager", "hr"}, "HR-BANDS": {"hr"}}


@pytest.mark.parametrize("employee_id", ["E1001", "E1003", "M2001", "H9001"])
@pytest.mark.parametrize(
    "query",
    [
        "Confidential approval matrix: managers grant 7 days of sick leave extensions.",
        "Confidential salary bands: band L3 pays 9 days of sick leave top-up.",
        "salary bands",
        "How many sick leave days do I get?",
    ],
)
def test_restricted_documents_never_reach_unauthorised_roles(retriever, db_conn, employee_id, query):
    """Must-pass #2: not in hits, not in rules, not in the context — even when the
    query is the restricted text itself."""
    role = get_employee(db_conn, employee_id).role
    result = ask(retriever, db_conn, employee_id, query=query)
    seen = {h.chunk_id.split("::")[0] for h in result.hits}
    seen |= {f.doc_id for r in result.resolutions for f in r.winners + r.outranked + r.upcoming}
    seen |= set(docs_in_context(result))
    for doc_id, roles in RESTRICTED.items():
        if role not in roles:
            assert doc_id not in seen
    assert result.context.tokens <= CORPUS_TOKEN_LIMIT


def test_authorised_role_can_retrieve_restricted_text(retriever, db_conn):
    result = ask(retriever, db_conn, "H9001", query="Confidential salary bands")
    assert result.hits[0].chunk_id.startswith("HR-BANDS::")


def test_app_loads_the_retriever_after_ingestion(client):
    assert client.app.state.app_state.retriever is not None
