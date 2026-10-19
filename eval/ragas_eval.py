"""RAGAS scores for the eval's document answers: faithfulness, answer relevancy,
context precision and context recall.

Usage (separate environment: RAGAS pins openai 1.x and langchain, the app pins openai 3.x):
    python -m venv .venv-eval
    .venv-eval/Scripts/python -m pip install -r requirements-eval.txt   (Windows; bin/ elsewhere)
    python eval/run_eval.py                       # app venv: writes eval/results.json
    .venv-eval/Scripts/python eval/ragas_eval.py  # run 1 only (default)
    .venv-eval/Scripts/python eval/ragas_eval.py --runs 1,2,3
    make ragas

Input: eval/results.json from run_eval.py. Per turn it holds what the composer was
given (requester details, corpus passages, tool results, notes: RAGAS's retrieved contexts) and the
case's `reference` answer from eval/cases.yaml. Scored: turns answered from documents
(route rag/hybrid) that have both; no_answer, refusals and tool-only turns are not.

Metrics (RAGAS 0.4, judged by the LLM in .env unless RAGAS_LLM_MODEL is set):
- faithfulness: share of the answer's statements supported by the contexts;
- answer relevancy: how well the answer addresses the question (questions generated
  from the answer, compared by embedding with the real question; local bge-small);
- context precision: are the contexts that support the reference ranked first;
- context recall: share of the reference's statements found in the contexts.

These complement the rule checks in eval/report.md, they don't replace them: a
passage from an outdated version is "relevant" to RAGAS, and an answer faithful to
it is still wrong. Version, region and access correctness stay with the checks.

Each judge call carries at most the contexts the composer got, so at most 3,000
corpus tokens (TASK.pdf §4.3); turns above that are skipped, never truncated.

Turns are scored a few at a time and saved to eval/.ragas_checkpoint.json after each
batch, so an interrupted run resumes where it stopped (a run on the free tier takes
about an hour). A turn whose judge call failed is retried on the next run.

Writes eval/ragas_report.md and eval/ragas_results.json.
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "eval" / "results.json"
REPORT_PATH = ROOT / "eval" / "ragas_report.md"
SCORES_PATH = ROOT / "eval" / "ragas_results.json"
CHECKPOINT_PATH = ROOT / "eval" / ".ragas_checkpoint.json"
BATCH_TURNS = 4
CORPUS_TOKEN_LIMIT = 3000
DOC_ROUTES = {"rag", "hybrid"}
# Column names RAGAS gives the four metrics, and how the report names them.
METRICS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy",
    "llm_context_precision_with_reference": "Context precision",
    "context_recall": "Context recall",
}
# Same variables as the app (app/config.py); OpenAI-compatible providers only.
_KEY_VARS = {
    "openai": "OPENAI_API_KEY", "nvidia": "NVIDIA_API_KEY", "gemini": "GOOGLE_API_KEY",
    "google": "GOOGLE_API_KEY", "openrouter": "OPENROUTER_API_KEY", "mistral": "MISTRAL_API_KEY",
}
_BASE_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openrouter": "https://openrouter.ai/api/v1",
    "mistral": "https://api.mistral.ai/v1",
}


def count_tokens(text: str) -> int:
    import tiktoken

    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def build_rows(results: list[dict], runs: set[int]) -> tuple[list[dict], dict[str, int]]:
    """RAGAS samples for the scored turns, plus counts of what was left out and why."""
    rows, skipped = [], {"not a document answer": 0, "no reference": 0, "no contexts": 0, "over 3,000 corpus tokens": 0}
    for r in results:
        if r["run"] not in runs:
            continue
        if r["body"].get("route") not in DOC_ROUTES:
            skipped["not a document answer"] += 1
        elif not r.get("reference"):
            skipped["no reference"] += 1
        elif not r.get("contexts"):
            skipped["no contexts"] += 1
        elif sum(count_tokens(c["text"]) for c in r["contexts"] if c["kind"] == "corpus") > CORPUS_TOKEN_LIMIT:
            skipped["over 3,000 corpus tokens"] += 1
        else:
            rows.append({
                "run": r["run"], "case": r["case"], "turn": r["turn"],
                "user_input": r["say"], "response": r["body"]["answer"],
                "retrieved_contexts": [c["text"] for c in r["contexts"]],
                "context_ids": [c["id"] for c in r["contexts"]],
                "reference": r["reference"],
            })
    return rows, skipped


def judge_llm():
    """The judge: the app's LLM settings from .env, or RAGAS_LLM_* to use another model."""
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    load_dotenv(ROOT / ".env")
    provider = os.environ.get("RAGAS_LLM_PROVIDER") or os.environ.get("LLM_PROVIDER", "openai")
    model = os.environ.get("RAGAS_LLM_MODEL") or os.environ.get("LLM_MODEL", "")
    if provider not in _KEY_VARS or not model:
        sys.exit(f"Set an OpenAI-compatible LLM_PROVIDER ({', '.join(_KEY_VARS)}) and LLM_MODEL in .env first.")
    base_url = os.environ.get("RAGAS_LLM_BASE_URL") or os.environ.get("LLM_BASE_URL") or _BASE_URLS.get(provider)
    extra = {"reasoning_effort": effort} if (effort := os.environ.get("LLM_REASONING_EFFORT")) else {}
    chat = ChatOpenAI(
        model=model, api_key=os.environ.get(_KEY_VARS[provider], ""), base_url=base_url,
        temperature=0, timeout=60, max_retries=2, **extra,
    )
    return LangchainLLMWrapper(chat), f"{provider}:{model}"


def local_embeddings():
    """bge-small-en-v1.5 on CPU, the app's own embedding model (app/llm/embeddings.py)."""
    from fastembed import TextEmbedding
    from ragas.embeddings import BaseRagasEmbeddings

    class FastEmbed(BaseRagasEmbeddings):
        def __init__(self) -> None:
            super().__init__()
            self.model = "BAAI/bge-small-en-v1.5"  # RAGAS logs this name with each embedding call
            self._encoder = TextEmbedding(self.model, cache_dir=os.environ.get("FASTEMBED_CACHE_PATH"))

        def embed_documents(self, texts):
            return [v.tolist() for v in self._encoder.embed(list(texts))]

        def embed_query(self, text):
            return next(iter(self._encoder.query_embed([text]))).tolist()

        async def aembed_documents(self, texts):
            return self.embed_documents(texts)

        async def aembed_query(self, text):
            return self.embed_query(text)

    return FastEmbed()


def row_key(row: dict) -> str:
    """Identifies a scored turn; changes when its answer, contexts or reference change."""
    content = json.dumps([row["response"], row["retrieved_contexts"], row["reference"]], ensure_ascii=False)
    return f"{row['run']}|{row['case']}|{row['turn']}|{hashlib.sha1(content.encode()).hexdigest()[:12]}"


def is_scored(row: dict) -> bool:
    return all(row.get(column) is not None for column in METRICS)


def score(rows: list[dict], workers: int, llm, embeddings) -> list[dict]:
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy
    from ragas.run_config import RunConfig

    dataset = EvaluationDataset.from_list(
        [{k: row[k] for k in ("user_input", "response", "retrieved_contexts", "reference")} for row in rows]
    )
    # strictness=1: one generated question per answer (some OpenAI-compatible endpoints ignore n > 1)
    metrics = [Faithfulness(), ResponseRelevancy(strictness=1), LLMContextPrecisionWithReference(), LLMContextRecall()]
    result = evaluate(
        dataset, metrics=metrics, llm=llm, embeddings=embeddings,
        # Context precision makes one judge call per context (10-20 per turn): 600 s per metric and turn.
        run_config=RunConfig(timeout=600, max_retries=3, max_workers=workers), raise_exceptions=False,
    )
    table = result.to_pandas()
    for row, (_, scores) in zip(rows, table.iterrows()):
        for column in METRICS:
            value = scores.get(column)
            row[column] = None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)
    return rows


def score_resumable(rows: list[dict], workers: int) -> tuple[list[dict], str]:
    """Score in batches of BATCH_TURNS, saving each batch; turns already fully scored
    in the checkpoint are reused."""
    try:
        saved = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    for row in rows:
        if is_scored(done := saved.get(row_key(row), {})):
            row.update({column: done[column] for column in METRICS})
    todo = [row for row in rows if not is_scored(row)]
    llm, judge = judge_llm()
    if todo:
        print(f"{len(rows) - len(todo)} turns already scored (checkpoint); scoring {len(todo)}", flush=True)
        embeddings = local_embeddings()
        for start in range(0, len(todo), BATCH_TURNS):
            batch = score(todo[start:start + BATCH_TURNS], workers, llm, embeddings)
            saved.update({row_key(row): {column: row[column] for column in METRICS} for row in batch})
            CHECKPOINT_PATH.write_text(json.dumps(saved, indent=1), encoding="utf-8")
            print(f"saved {min(start + BATCH_TURNS, len(todo))}/{len(todo)}", flush=True)
    return rows, judge


def summary(rows: list[dict], column: str) -> tuple[float | None, int]:
    values = [r[column] for r in rows if r.get(column) is not None]
    return (statistics.mean(values) if values else None), len(values)


def write_report(rows: list[dict], skipped: dict[str, int], judge: str, runs: list[int]) -> str:
    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2f}"

    lines = [
        "# RAGAS report",
        "",
        f"Judge: `{judge}` · runs scored: {', '.join(map(str, runs))} · turns scored: {len(rows)}",
        "",
        "Generated by `eval/ragas_eval.py` from `eval/results.json`. Scores are 0 to 1, higher is better.",
        "They are judged by an LLM, so they vary a little between runs; the pass/fail grade is the",
        "rule checks in `eval/report.md` (exact values, versions, regions, access, tools).",
        "",
        "## Scores",
        "",
        "| Metric | Mean | Turns scored |",
        "|---|---|---|",
    ]
    for column, name in METRICS.items():
        mean, n = summary(rows, column)
        lines.append(f"| {name} | {fmt(mean)} | {n}/{len(rows)} |")
    left_out = ", ".join(f"{why}: {n}" for why, n in skipped.items() if n) or "none"
    lines += ["", f"Not scored: {left_out}. A turn whose judge call failed shows n/a.", "",
              "## By case", "", "| Case | " + " | ".join(METRICS.values()) + " |", "|---|" + "---|" * len(METRICS)]
    for case in dict.fromkeys(r["case"] for r in rows):
        case_rows = [r for r in rows if r["case"] == case]
        lines.append(f"| `{case}` | " + " | ".join(fmt(summary(case_rows, c)[0]) for c in METRICS) + " |")
    lines += ["", "## Lowest scores", ""]
    for column, name in METRICS.items():
        low = sorted((r for r in rows if r.get(column) is not None and r[column] < 0.5), key=lambda r: r[column])[:5]
        lines += [f"### {name}", ""]
        if not low:
            lines += ["No turn below 0.50.", ""]
            continue
        for r in low:
            lines += [
                f"- `{r['case']}` (run {r['run']}): **{r[column]:.2f}** · contexts: {', '.join(r['context_ids'])}",
                f"  - question: {r['user_input']}",
                f"  - answer: {r['response'][:300]}",
                f"  - reference: {r['reference']}",
            ]
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="1", help="comma-separated run numbers from results.json (default 1)")
    parser.add_argument("--workers", type=int, default=4, help="parallel judge calls (free tiers: keep low)")
    args = parser.parse_args()
    runs = sorted(int(n) for n in args.runs.split(","))

    if not RESULTS_PATH.exists():
        sys.exit("Run eval/run_eval.py first (it writes eval/results.json).")
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    if not any("contexts" in r for r in results):
        sys.exit("eval/results.json has no contexts: rerun eval/run_eval.py (older runs didn't save them).")
    rows, skipped = build_rows(results, set(runs))
    if not rows:
        sys.exit(f"Nothing to score (left out: {skipped}).")
    print(f"scoring {len(rows)} turns with RAGAS ({len(METRICS)} metrics)...", flush=True)
    rows, judge = score_resumable(rows, args.workers)
    REPORT_PATH.write_text(write_report(rows, skipped, judge, runs), encoding="utf-8")
    SCORES_PATH.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    for column, name in METRICS.items():
        mean, n = summary(rows, column)
        print(f"{name:18} {'n/a' if mean is None else f'{mean:.2f}'}  ({n}/{len(rows)} scored)")
    print(f"\nwrote {REPORT_PATH.relative_to(ROOT)} and {SCORES_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
