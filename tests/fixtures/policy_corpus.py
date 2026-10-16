"""A tiny policy corpus mirroring the spec's scenario, plus a fake fact-extraction
model. Shared by the retrieval and chat tests. All content is invented."""

import json
import re
from pathlib import Path

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


def write_corpus(root: Path) -> tuple[Path, Path]:
    """Writes corpus/ and manifest.json under `root`; returns (corpus_dir, manifest_path)."""
    corpus = root / "corpus"
    corpus.mkdir()
    manifest = []
    for doc_id, (doc_type, regions, effective, access, sentence) in DOCS.items():
        (corpus / f"{doc_id}.md").write_text(f"# {doc_id}\n\n## Sick leave\n\n{sentence}\n", encoding="utf-8")
        manifest.append({"doc_id": doc_id, "file": f"{doc_id}.md", "title": doc_id, "doc_type": doc_type,
                         "regions": regions, "effective_date": effective, "access_level": access})
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return corpus, manifest_path
