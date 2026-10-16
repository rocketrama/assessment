"""OCR for PDF pages without a text layer (ARCHITECTURE §5.3).

The page is rendered at 300 DPI. A vision model transcribes it to markdown when
one is configured; Tesseract reads it too, and the numbers and dates in the two
outputs are compared — a vision number Tesseract didn't see flags the page as
low-confidence. Tesseract is the fallback when the vision call fails, so a
provider outage can't keep /health at 503.

OCR text is untrusted corpus text like any other. Headings are not taken from
OCR output: scanned pages are cited by page number (`p. N`).
"""

import io
import logging
import re
import shutil
from dataclasses import dataclass, field
from typing import Protocol

import pypdfium2

from app.ingestion.blocks import Block

log = logging.getLogger("veltrix.ingestion")

RENDER_DPI = 300
_NUMBER_RE = re.compile(r"\d+")


class Transcriber(Protocol):
    """Image (JPEG bytes) → verbatim markdown. Implemented by app/llm/vision.py."""

    model: str

    def transcribe(self, image_jpeg: bytes) -> str: ...


def tesseract_available(cmd: str = "") -> bool:
    """True when Tesseract runs. `cmd` is an explicit binary path (TESSERACT_CMD)."""
    import pytesseract

    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    elif shutil.which("tesseract") is None:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


@dataclass
class PageOcr:
    vision: Transcriber | None = None
    use_tesseract: bool = field(default_factory=tesseract_available)
    min_confidence: float = 70.0

    def engines(self) -> str:
        """Part of the cache key: a different engine setup must re-run OCR."""
        return f"vision={self.vision.model if self.vision else '-'};tesseract={self.use_tesseract}"

    def ocr_pdf_page(self, path: str, page_no: int) -> tuple[list[Block], list[str], bool]:
        """(blocks, warnings, degraded). Degraded = the best configured engine didn't
        produce this result, so it must not be cached."""
        image = _render(path, page_no)
        warnings: list[str] = []

        tess_text, tess_conf = None, None
        if self.use_tesseract:
            try:
                tess_text, tess_conf = _tesseract(image)
            except Exception as exc:
                warnings.append(f"page {page_no}: tesseract failed ({type(exc).__name__})")

        if self.vision is not None:
            try:
                text = _strip_fences(self.vision.transcribe(_jpeg_bytes(image)))
                low = False
                if tess_text is not None:
                    missing = set(_NUMBER_RE.findall(text)) - set(_NUMBER_RE.findall(tess_text))
                    if missing:
                        low = True
                        warnings.append(
                            f"page {page_no}: vision numbers not confirmed by tesseract: {sorted(missing)[:10]}"
                        )
                return _ocr_blocks(text, page_no, "vision", low), warnings, False
            except Exception as exc:
                warnings.append(f"page {page_no}: vision OCR failed ({type(exc).__name__}); falling back")

        if tess_text is not None:
            low = tess_conf is not None and tess_conf < self.min_confidence
            if low:
                warnings.append(f"page {page_no}: tesseract confidence {tess_conf:.0f} below threshold")
            return _ocr_blocks(tess_text, page_no, "tesseract", low), warnings, self.vision is not None

        warnings.append(f"page {page_no}: no text layer and no OCR engine succeeded")
        return [], warnings, True


def _render(path: str, page_no: int):
    pdf = pypdfium2.PdfDocument(path)
    try:
        return pdf[page_no - 1].render(scale=RENDER_DPI / 72).to_pil().convert("RGB")
    finally:
        pdf.close()


def _jpeg_bytes(image, max_side: int = 1600) -> bytes:
    # Small API payload (~100–200 KB instead of >1 MB as PNG); 1600 px grayscale
    # keeps body text sharp. Tesseract still reads the full 300 DPI render.
    scale = max_side / max(image.size)
    if scale < 1:
        image = image.resize((round(image.width * scale), round(image.height * scale)))
    buf = io.BytesIO()
    image.convert("L").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _tesseract(image) -> tuple[str, float | None]:
    import pytesseract

    text = pytesseract.image_to_string(image)
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    confs = [float(c) for c, w in zip(data["conf"], data["text"]) if w.strip() and float(c) >= 0]
    return text, (sum(confs) / len(confs) if confs else None)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _ocr_blocks(text: str, page_no: int, engine: str, low_confidence: bool) -> list[Block]:
    from app.ingestion.loaders.markdown import parse_markdown_text

    out = []
    for b in parse_markdown_text(text, page=page_no):
        kind = "text" if b.kind == "heading" else b.kind
        out.append(
            Block(kind, b.text, page=page_no, rows=b.rows, ocr_engine=engine, low_confidence=low_confidence)
        )
    return out
