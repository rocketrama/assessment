"""Show what retrieval does for one question, as one employee — no LLM call.

Usage:
    python scripts/ask_retrieval.py E1001 "How many sick days do I get?"
    python scripts/ask_retrieval.py E1001 "What was the sick leave allowance?" --date 2025-08-01

Reads the ingestion cache (start the app once first) and veltrix.db from .env.
Prints the eligible-chunk count, the top hits, each rule's winner / conflict /
upcoming change, and the packed context with its corpus-token count.
"""

import argparse
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Settings  # noqa: E402
from app.data.db import connect  # noqa: E402
from app.data.employees import get_employee  # noqa: E402
from app.ingestion.cache import cache_db_path  # noqa: E402
from app.llm.embeddings import Embedder  # noqa: E402
from app.retrieval.retriever import Retriever  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("employee_id")
    parser.add_argument("question")
    parser.add_argument("--date", help="reference date YYYY-MM-DD (default: APP_TODAY)")
    args = parser.parse_args()

    settings = Settings()
    with connect(settings.db_path) as conn:
        employee = get_employee(conn, args.employee_id)
    if employee is None:
        sys.exit(f"unknown employee {args.employee_id}")
    reference = date.fromisoformat(args.date) if args.date else settings.today()

    retriever = Retriever.load(cache_db_path(settings.cache_dir), Embedder())
    corpus = retriever.corpus
    result = retriever.retrieve(args.question, employee, reference, today=settings.today())

    print(f"{employee.employee_id} role={employee.role} region={employee.region} reference={reference}")
    print(f"eligible chunks: {result.allowed_chunks} of {len(corpus.chunks)}\n")
    print("top hits:")
    for h in result.hits:
        c = corpus.chunks[h.chunk_id]
        print(f"  {h.chunk_id:32} bm25#{h.bm25_rank or '-':<3} dense#{h.dense_rank:<3} {c.locator}")
    print("\nrules:")
    for r in result.resolutions:
        status = "CONFLICT" if r.conflict else "winner"
        win = ", ".join(f"{f.value} ({f.doc_id})" for f in r.winners) or "-"
        print(f"  [{r.rule_key}] {status}: {win}")
        if r.outranked:
            print("      outranked: " + ", ".join(f"{f.value} ({f.doc_id})" for f in r.outranked))
        for f in r.upcoming:
            print(f"      upcoming: {f.value} ({f.doc_id}, from {corpus.docs[f.doc_id].effective_date})")
    print(f"\ncontext: {len(result.context.segments)} passages, {result.context.tokens} corpus tokens "
          f"(limit 3000), omitted {len(result.context.omitted)}")
    for s in result.context.segments:
        print(f"  {s.chunk_id:32} {corpus.chunks[s.chunk_id].locator}")


if __name__ == "__main__":
    main()
