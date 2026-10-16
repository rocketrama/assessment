"""manifest.json validation: bad entries are skipped with a reason, never fatal (TASK.pdf §2.2)."""

import json

import pytest

from app.ingestion.manifest import ManifestError, load_manifest

GOOD = {
    "doc_id": "POL-A",
    "file": "a.md",
    "title": "A",
    "doc_type": "policy",
    "regions": ["global"],
    "effective_date": "2026-01-01",
    "access_level": "all",
}


def _write(tmp_path, entries, files=("a.md",)):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name in files:
        (corpus / name).write_text("# A", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(entries), encoding="utf-8")
    return str(manifest), str(corpus)


def test_valid_entry(tmp_path):
    entries, skipped = load_manifest(*_write(tmp_path, [GOOD]))
    assert skipped == []
    (e,) = entries
    assert (e.doc_id, e.regions, e.effective_date.isoformat()) == ("POL-A", ("GLOBAL",), "2026-01-01")


@pytest.mark.parametrize(
    "bad, reason",
    [
        ({**GOOD, "doc_type": "rumour"}, "invalid fields: doc_type"),
        ({**GOOD, "access_level": "everyone"}, "invalid fields: access_level"),
        ({**GOOD, "effective_date": "01/02/2026"}, "invalid fields: effective_date"),
        ({**GOOD, "regions": []}, "invalid fields: regions"),
        ({**GOOD, "file": "../secrets.md"}, "invalid fields: file"),
        ({**GOOD, "file": "sub/a.md"}, "invalid fields: file"),
        ({**GOOD, "file": "missing.md"}, "file not found in corpus"),
        ({**GOOD, "file": "a.txt"}, "unsupported file type"),
    ],
)
def test_bad_entries_are_skipped_with_reason(tmp_path, bad, reason):
    entries, skipped = load_manifest(*_write(tmp_path, [bad], files=("a.md", "a.txt")))
    assert entries == []
    assert [s.reason for s in skipped] == [reason]


def test_duplicate_doc_id_keeps_first(tmp_path):
    entries, skipped = load_manifest(*_write(tmp_path, [GOOD, {**GOOD, "title": "B"}]))
    assert [e.title for e in entries] == ["A"]
    assert skipped[0].reason == "duplicate doc_id"


def test_non_object_entry_is_skipped(tmp_path):
    entries, skipped = load_manifest(*_write(tmp_path, ["not an object", GOOD]))
    assert len(entries) == 1 and skipped[0].index == 0


@pytest.mark.parametrize("content", ["{}", "not json"])
def test_unusable_manifest_raises(tmp_path, content):
    manifest, corpus = _write(tmp_path, [])
    with open(manifest, "w", encoding="utf-8") as f:
        f.write(content)
    with pytest.raises(ManifestError):
        load_manifest(manifest, corpus)
