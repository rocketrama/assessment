"""OCR for pages without a text layer: vision first, Tesseract cross-check and
fallback (ARCHITECTURE §5.3). Engines are faked, so no API key or Tesseract
install is needed."""

import pytest

from app.ingestion.loaders import load_document
from app.ingestion.loaders import ocr as ocr_module
from app.ingestion.loaders.ocr import PageOcr
from tests.fixtures.make_pdf import sample_policy_pdf

VISION_TEXT = "```markdown\n# MEMO HEADER\n\nWork from home up to 3 days per week.\n\n| Office | Days |\n|---|---|\n| X | 3 |\n```"


class FakeVision:
    model = "fake-vision"

    def __init__(self, text=VISION_TEXT, fail=False):
        self.text, self.fail, self.calls = text, fail, 0

    def transcribe(self, image_jpeg: bytes) -> str:
        self.calls += 1
        assert image_jpeg.startswith(b"\xff\xd8")  # JPEG
        assert len(image_jpeg) < 400_000
        if self.fail:
            raise TimeoutError("provider timeout")
        return self.text


@pytest.fixture
def scanned_pdf(tmp_path):
    path = tmp_path / "scan.pdf"
    path.write_bytes(sample_policy_pdf(with_scanned_page=True))
    return str(path)


def _tesseract_returns(monkeypatch, text, conf=90.0):
    monkeypatch.setattr(ocr_module, "_tesseract", lambda image: (text, conf))


def _page2(doc):
    return [b for b in doc.blocks if b.page == 2]


def test_vision_transcript_becomes_page_blocks_without_headings(scanned_pdf):
    vision = FakeVision()
    doc = load_document(scanned_pdf, PageOcr(vision=vision, use_tesseract=False))
    blocks = _page2(doc)
    assert vision.calls == 1
    assert [(b.kind, b.text) for b in blocks] == [
        ("text", "MEMO HEADER"),  # OCR headings are not trusted as sections
        ("text", "Work from home up to 3 days per week."),
        ("table", "| Office | Days |\n|---|---|\n| X | 3 |"),
    ]
    assert all(b.ocr_engine == "vision" and not b.low_confidence for b in blocks)
    assert not doc.degraded


def test_vision_numbers_cross_checked_against_tesseract(scanned_pdf, monkeypatch):
    _tesseract_returns(monkeypatch, "Work from home up to 8 days per week. Office X 8")
    doc = load_document(scanned_pdf, PageOcr(vision=FakeVision(), use_tesseract=True))
    assert all(b.low_confidence for b in _page2(doc))
    assert any("not confirmed by tesseract" in w for w in doc.warnings)


def test_vision_failure_falls_back_to_tesseract_and_is_not_cacheable(scanned_pdf, monkeypatch):
    _tesseract_returns(monkeypatch, "Work from home up to 3 days per week.")
    doc = load_document(scanned_pdf, PageOcr(vision=FakeVision(fail=True), use_tesseract=True))
    assert [(b.text, b.ocr_engine) for b in _page2(doc)] == [("Work from home up to 3 days per week.", "tesseract")]
    assert doc.degraded  # retry vision next startup
    assert any("vision OCR failed (TimeoutError)" in w for w in doc.warnings)


def test_tesseract_only_is_a_normal_result(scanned_pdf, monkeypatch):
    _tesseract_returns(monkeypatch, "Plain scanned text.", conf=55.0)
    doc = load_document(scanned_pdf, PageOcr(vision=None, use_tesseract=True, min_confidence=70))
    (block,) = _page2(doc)
    assert (block.ocr_engine, block.low_confidence) == ("tesseract", True)
    assert not doc.degraded


def test_no_engine_leaves_page_empty_and_degraded(scanned_pdf):
    doc = load_document(scanned_pdf, PageOcr(vision=None, use_tesseract=False))
    assert _page2(doc) == [] and doc.degraded


def test_engines_string_is_part_of_cache_identity():
    assert PageOcr(vision=None, use_tesseract=False).engines() != PageOcr(vision=FakeVision(), use_tesseract=False).engines()
