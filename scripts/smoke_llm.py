"""Pick an LLM model by evidence: JSON validity, consistency and latency.

Usage:
    python scripts/smoke_llm.py <model_id> [<model_id> ...]

Reads LLM_BASE_URL and the provider key from .env. Runs a planner-style
JSON task 3 times per model at temperature 0 and reports:
  - valid:      replies that parse as the expected JSON object
  - clean:      replies with no text outside the JSON (e.g. "thinking" output)
  - consistent: all valid replies identical
  - latency:    seconds per call (each /chat must finish within 30 s)
"""

import json
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

RUNS = 3
EXPECTED_KEYS = {"intent", "leave_type", "start_date", "end_date", "missing"}

SYSTEM = (
    "You extract structured intent from an employee's HR message. "
    "Reply with ONLY a JSON object, no other text, with keys: "
    'intent (one of "policy_question", "leave_write", "leave_read", "off_topic"), '
    "leave_type (\"annual\", \"sick\" or null), start_date and end_date "
    "(YYYY-MM-DD or null), missing (list of missing field names). "
    "Today is 2026-09-15 (Tuesday). A date without a year means 2026."
)
USER = "I'd like to take leave from 21 September to 23 September."


def call(client: OpenAI, model: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=300,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}],
    )
    return resp.choices[0].message.content or "", time.perf_counter() - t0


def parse(text: str) -> tuple[dict | None, bool]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").removeprefix("json").strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start == -1 or end == -1:
        return None, False
    clean = start == 0 and end == len(stripped) - 1
    try:
        obj = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None, clean
    return (obj if EXPECTED_KEYS <= obj.keys() else None), clean


def main() -> None:
    load_dotenv()
    models = sys.argv[1:]
    if not models:
        sys.exit(__doc__)
    key = os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        sys.exit("Set NVIDIA_API_KEY (or OPENAI_API_KEY) in .env")
    client = OpenAI(base_url=os.getenv("LLM_BASE_URL") or None, api_key=key, timeout=60)

    for model in models:
        results, latencies, cleans, errors = [], [], [], []
        for _ in range(RUNS):
            try:
                text, secs = call(client, model)
            except Exception as exc:  # report and continue with the next model
                errors.append(type(exc).__name__ + ": " + str(exc)[:120])
                continue
            obj, clean = parse(text)
            results.append(obj)
            latencies.append(secs)
            cleans.append(clean)
        valid = [r for r in results if r is not None]
        consistent = len(valid) == RUNS and all(v == valid[0] for v in valid)
        print(f"\n== {model}")
        if errors:
            print("   errors:", errors)
        if latencies:
            print(f"   valid {len(valid)}/{RUNS} | clean {sum(cleans)}/{RUNS} | consistent {consistent} "
                  f"| latency avg {sum(latencies) / len(latencies):.1f}s max {max(latencies):.1f}s")
        if valid:
            print("   sample:", json.dumps(valid[0]))


if __name__ == "__main__":
    main()
