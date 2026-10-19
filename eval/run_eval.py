"""Run eval/cases.yaml against the real service and model, several times, and score it.

Usage:
    python eval/run_eval.py                 # 3 runs (TASK.pdf §10.3: tests run 3 times)
    python eval/run_eval.py --runs 1 --only sick-days-india,carry-deadline-india
    make eval

Each run starts the app on a fresh copy of ./data (leave writes change the DB)
and a copy of ./cache (warm start), so ./data and ./cache are never changed.
Writes eval/report.md (scorecard, consistency, every failure with its reason)
and eval/results.json (raw responses, plus what the composer was given per turn
and the case's reference answer, for the RAGAS scores in eval/ragas_eval.py).
"""

import argparse
import json
import logging
import shutil
import sqlite3
import statistics
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.orchestration import orchestrator  # noqa: E402

CASES_PATH = ROOT / "eval" / "cases.yaml"
REPORT_PATH = ROOT / "eval" / "report.md"
RESULTS_PATH = ROOT / "eval" / "results.json"


def norm(text: str) -> str:
    """Case, whitespace and dash variants don't decide a match."""
    for dash in "‐‑‒–—":
        text = text.replace(dash, "-")
    return " ".join(text.lower().split())


# --- scoring one turn -----------------------------------------------------------------

def score_turn(expect: dict, body: dict) -> dict:
    """Checks for one response: {check name: (ok, detail)} plus precision/recall numbers."""
    checks: dict[str, tuple[bool, str]] = {}
    answer = norm(body.get("answer", ""))
    cited = {c["doc_id"] for c in body.get("citations", [])}
    route = body.get("route")

    if "route" in expect:
        checks["route"] = (route in expect["route"], f"got {route}, want {expect['route']}")
    must = set(expect.get("cite_all", []))
    if must:
        checks["cite_all"] = (must <= cited, f"missing {sorted(must - cited)}")
    any_of = set(expect.get("cite_any", []))
    if any_of:
        checks["cite_any"] = (bool(any_of & cited), f"none of {sorted(any_of)} cited")
    banned = set(expect.get("cite_none", []))
    if banned:
        checks["cite_none"] = (not (banned & cited), f"cited {sorted(banned & cited)}")
    for item in expect.get("answer_has", []):
        options = item if isinstance(item, list) else [item]
        checks[f"has:{options[0]}"] = (any(norm(o) in answer for o in options), f"answer lacks {options}")
    for item in expect.get("answer_lacks", []):
        checks[f"lacks:{item}"] = (norm(item) not in answer, f"answer contains {item!r}")
    if "pending" in expect:
        checks["pending"] = (body.get("pending_confirmation") == expect["pending"], f"pending={body.get('pending_confirmation')}")
    if "tool" in expect:
        want = expect["tool"]
        calls = [t for t in body.get("tool_calls", []) if t["name"] == want["name"]]
        ok = any(
            (want.get("status") in (None, t["status"]))
            and all(str(t["args"].get(k)) == str(v) for k, v in want.get("args", {}).items())
            for t in calls
        )
        checks["tool"] = (ok, f"want {want}, got {body.get('tool_calls')}")

    expected_docs = must | any_of | set(expect.get("cite_ok", []))
    precision = (
        len(cited & expected_docs) / len(cited) if cited and expected_docs else None
    )  # only when the case says which documents are right
    recall = len(must & cited) / len(must) if must else None
    return {"checks": checks, "precision": precision, "recall": recall}


# --- running -----------------------------------------------------------------------------

def make_settings(work: Path, base: Settings) -> Settings:
    data = work / "data"
    shutil.copytree(base.corpus_dir, data / "corpus")
    shutil.copy(base.manifest_path, data / "manifest.json")
    src, dst = sqlite3.connect(base.db_path), sqlite3.connect(data / "veltrix.db")
    src.backup(dst)
    src.close()
    dst.close()
    if Path(base.cache_dir).exists():
        shutil.copytree(base.cache_dir, work / "cache")
    return base.model_copy(update={
        "corpus_dir": str(data / "corpus"), "manifest_path": str(data / "manifest.json"),
        "db_path": str(data / "veltrix.db"), "cache_dir": str(work / "cache"), "log_level": "WARNING",
    })


class ComposerInputs:
    """What the composer was given for the current turn: the corpus passages, tool
    results and notes. Saved per turn as RAGAS's `retrieved_contexts` (eval/ragas_eval.py)."""

    def __init__(self) -> None:
        self.contexts: list[dict] = []
        self._compose = orchestrator.compose

    def __enter__(self) -> "ComposerInputs":
        def compose(llm, message, requester, today, reference_date, segments, tool_results, notes, *a, **k):
            if self.contexts:  # a rewrite pass: same passages, plus an instruction that isn't context
                return self._compose(llm, message, requester, today, reference_date, segments, tool_results, notes, *a, **k)
            # The requester details the composer is given (level, join date, region): RAGAS's
            # judge needs them too, or "your notice period is 30 days" (L1-L3) looks unsupported.
            about = {"today": today.isoformat(), "employee_id": requester.employee_id, "role": requester.role,
                     "region": requester.region, "level": requester.level, "join_date": requester.join_date}
            if reference_date and reference_date != today:
                about["question_is_about_date"] = reference_date.isoformat()
            self.contexts = [{"id": "requester", "kind": "requester", "text": "Requester: " + json.dumps(about)}]
            self.contexts += [{"id": s.chunk_id, "kind": "corpus", "text": s.text} for s in segments]
            self.contexts += [{"id": f"tool:{t['tool']}", "kind": "tool", "text": json.dumps(t, default=str)} for t in tool_results]
            self.contexts += [{"id": "note", "kind": "note", "text": n} for n in notes]
            return self._compose(llm, message, requester, today, reference_date, segments, tool_results, notes, *a, **k)

        orchestrator.compose = compose
        return self

    def __exit__(self, *exc) -> None:
        orchestrator.compose = self._compose


class NoAnswerReasons(logging.Handler):
    """Collects the orchestrator's "no_answer" log records (reason only, no text)."""

    def __init__(self) -> None:
        super().__init__()
        self.reasons: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage() == "no_answer":
            self.reasons.append(getattr(record, "reason", "?"))


def run_once(cases: list[dict], base: Settings, run_no: int) -> list[dict]:
    work = Path(tempfile.mkdtemp(prefix=f"eval-run{run_no}-"))
    results = []
    try:
        with TestClient(create_app(make_settings(work, base))) as client, ComposerInputs() as inputs:
            started = time.monotonic()
            while client.get("/health").status_code != 200:
                if time.monotonic() - started > 600:
                    raise RuntimeError("service not ready after 10 minutes")
                time.sleep(1)
            reasons = NoAnswerReasons()
            orchestration_log = logging.getLogger("veltrix.orchestration")
            orchestration_log.setLevel(logging.INFO)
            orchestration_log.propagate = False
            orchestration_log.addHandler(reasons)
            for case in cases:
                session = f"eval-{run_no}-{case['id']}-{time.time_ns()}"
                for i, turn in enumerate(case["turns"]):
                    reasons.reasons.clear()
                    inputs.contexts = []
                    t0 = time.monotonic()
                    resp = client.post("/chat", json={"session_id": session, "message": turn["say"]},
                                       headers={"X-Employee-Id": case["employee"]})
                    seconds = time.monotonic() - t0
                    body = resp.json() if resp.status_code == 200 else {"route": f"HTTP {resp.status_code}"}
                    scored = score_turn(turn, body)
                    results.append({
                        "run": run_no, "case": case["id"], "turn": i, "employee": case["employee"],
                        "say": turn["say"], "seconds": round(seconds, 2), "body": body,
                        "ok": all(ok for ok, _ in scored["checks"].values()),
                        "failures": [f"{k}: {d}" for k, (ok, d) in scored["checks"].items() if not ok],
                        "checks": {k: ok for k, (ok, _) in scored["checks"].items()},
                        "precision": scored["precision"], "recall": scored["recall"],
                        "no_answer_reason": reasons.reasons[-1] if reasons.reasons else None,
                        "reference": turn.get("reference"), "contexts": inputs.contexts,
                    })
                    print(f"run {run_no} {case['id']}[{i}] {'PASS' if results[-1]['ok'] else 'FAIL'} "
                          f"{body.get('route')} {seconds:.1f}s", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return results


# --- report -------------------------------------------------------------------------------

def signature(r: dict) -> tuple:
    """What must not change between runs: route, cited docs, tools, and which checks pass."""
    b = r["body"]
    return (
        b.get("route"),
        tuple(sorted({c["doc_id"] for c in b.get("citations", [])})),
        tuple((t["name"], t["status"]) for t in b.get("tool_calls", [])),
        tuple(sorted(r["checks"].items())),
    )


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "n/a"


def write_report(cases: list[dict], results: list[dict], runs: int, model: str) -> str:
    by_case: dict[str, list[dict]] = {}
    for r in results:
        by_case.setdefault(r["case"], []).append(r)

    def check_rate(name_prefix: str) -> str:
        values = [ok for r in results for k, ok in r["checks"].items() if k.startswith(name_prefix)]
        return pct(sum(values), len(values))

    precisions = [r["precision"] for r in results if r["precision"] is not None]
    recalls = [r["recall"] for r in results if r["recall"] is not None]
    seconds = sorted(r["seconds"] for r in results)
    turn_keys = sorted({(r["case"], r["turn"]) for r in results})
    consistent = [
        k for k in turn_keys
        if len({signature(r) for r in results if (r["case"], r["turn"]) == k}) == 1
    ]
    case_pass = {
        cid: sum(all(r["ok"] for r in rs if r["run"] == run) for run in range(1, runs + 1))
        for cid, rs in by_case.items()
    }
    tags = Counter(t for c in cases for t in c.get("tags", []))
    tag_rows = []
    for tag in sorted(tags):
        ids = [c["id"] for c in cases if tag in c.get("tags", [])]
        passed = sum(case_pass[i] for i in ids)
        tag_rows.append(f"| {tag} | {len(ids)} | {pct(passed, len(ids) * runs)} |")

    lines = [
        "# Eval report",
        "",
        f"Model: `{model}` · runs: {runs} · cases: {len(cases)} · turns per run: {len(turn_keys)}",
        "",
        "Generated by `eval/run_eval.py` from `eval/cases.yaml` against the real service and model.",
        "",
        "## Scorecard",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Cases fully correct (all runs) | {pct(sum(case_pass.values()), len(cases) * runs)} |",
        f"| Route accuracy | {check_rate('route')} |",
        f"| Required citations present (recall) | {statistics.mean(recalls):.0%} over {len(recalls)} turns |" if recalls else "| Recall | n/a |",
        f"| Citation precision (cited docs that were expected) | {statistics.mean(precisions):.0%} over {len(precisions)} turns |" if precisions else "| Precision | n/a |",
        f"| Forbidden citations avoided (outdated / restricted) | {check_rate('cite_none')} |",
        f"| Key facts present in the answer | {check_rate('has:')} |",
        f"| Wrong or forbidden content avoided | {check_rate('lacks:')} |",
        f"| Tool calls as expected | {check_rate('tool')} |",
        f"| **Consistent across {runs} runs** (route, citations, tools, checks) | {pct(len(consistent), len(turn_keys))} |",
        f"| Latency median / p95 / max | {statistics.median(seconds):.1f} s / {seconds[int(0.95 * (len(seconds) - 1))]:.1f} s / {seconds[-1]:.1f} s (limit 30 s) |",
        "",
        "## By tag",
        "",
        "| Tag | Cases | Fully correct (case × run) |",
        "|---|---|---|",
        *tag_rows,
        "",
        "## Cases",
        "",
        "| Case | Passed runs | Consistent | Failures (first run that failed) |",
        "|---|---|---|---|",
    ]
    for case in cases:
        rs = by_case[case["id"]]
        cons = all(k in consistent for k in turn_keys if k[0] == case["id"])
        first_fail = next((r for r in rs if not r["ok"]), None)
        detail = "; ".join(first_fail["failures"])[:220] if first_fail else ""
        lines.append(f"| `{case['id']}` | {case_pass[case['id']]}/{runs} | {'yes' if cons else '**no**'} | {detail} |")
    lines += ["", "## Failed answers", ""]
    shown = set()
    for r in results:
        if r["ok"] or (r["case"], r["turn"]) in shown:
            continue
        shown.add((r["case"], r["turn"]))
        cites = ", ".join(sorted({c["doc_id"] for c in r["body"].get("citations", [])})) or "none"
        lines += [
            f"### `{r['case']}` turn {r['turn']} (run {r['run']}, {r['employee']})",
            f"> {r['say']}",
            "",
            f"- route: `{r['body'].get('route')}` · cited: {cites}"
            + (f" · no_answer reason: {r['no_answer_reason']}" if r.get("no_answer_reason") else ""),
            f"- failed: {'; '.join(r['failures'])}",
            f"- answer: {r['body'].get('answer', '')[:500]}",
            "",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--only", help="comma-separated case ids")
    args = parser.parse_args()

    cases = yaml.safe_load(CASES_PATH.read_text(encoding="utf-8"))
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    base = Settings()
    if not base.llm_model:
        sys.exit("Set LLM_PROVIDER, LLM_MODEL and the API key in .env first.")

    results = []
    for run in range(1, args.runs + 1):
        results += run_once(cases, base, run)
    REPORT_PATH.write_text(write_report(cases, results, args.runs, f"{base.llm_provider}:{base.llm_model}"), encoding="utf-8")
    RESULTS_PATH.write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    print(f"\nwrote {REPORT_PATH.relative_to(ROOT)} and {RESULTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
