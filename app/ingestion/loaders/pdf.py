"""PDF loader (pdfplumber). Headings are found from font size and weight, tables from
ruling lines, and pages without a text layer are sent to OCR (ARCHITECTURE §5.3).

Two passes: the first collects every line and table per page; the second knows
the document's body font size, so it can tell headings (bigger and bold) from
body text and drop the small running footer at the bottom of each page.
"""

from collections import Counter
from dataclasses import dataclass

import pdfplumber

from app.ingestion.blocks import Block, ParsedDocument, clean_text, is_metadata_table, table_block
from app.ingestion.loaders.ocr import PageOcr

MIN_TEXT_CHARS = 20  # fewer visible characters than this → treat the page as scanned
HEADING_RATIO = 1.15  # heading font ≥ 115% of the body font
CHROME_RATIO = 0.9  # running header/footer font < 90% of the body font
CHROME_MARGIN = 0.08  # ... and within the top/bottom 8% of the page


@dataclass
class _Line:
    top: float
    bottom: float
    size: float
    bold: bool
    text: str


@dataclass
class _Table:
    top: float
    rows: list[list[str]]


def _lines(page, table_boxes) -> list[_Line]:
    out = []
    for ln in page.extract_text_lines(return_chars=True):
        chars = [c for c in ln["chars"] if not c["text"].isspace()]
        if not chars:
            continue
        mid_x = (ln["x0"] + ln["x1"]) / 2
        mid_y = (ln["top"] + ln["bottom"]) / 2
        if any(x0 <= mid_x <= x1 and top <= mid_y <= bottom for x0, top, x1, bottom in table_boxes):
            continue
        size = Counter(round(c["size"], 1) for c in chars).most_common(1)[0][0]
        bold = sum("bold" in c["fontname"].lower() for c in chars) > len(chars) / 2
        out.append(_Line(ln["top"], ln["bottom"], size, bold, ln["text"]))
    return out


def load_pdf(path: str, ocr: PageOcr | None = None) -> ParsedDocument:
    doc = ParsedDocument()
    pages: list[tuple[int, list[_Line], list[_Table], float]] = []
    size_weights: Counter[float] = Counter()

    with pdfplumber.open(path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            visible = sum(1 for c in page.chars if not c["text"].isspace())
            if visible < MIN_TEXT_CHARS:
                pages.append((page_no, [], [], 0.0))  # placeholder; OCR below keeps page order
                continue
            found = page.find_tables()
            tables = [_Table(t.bbox[1], t.extract()) for t in found]
            lines = _lines(page, [t.bbox for t in found])
            for ln in lines:
                size_weights[ln.size] += len(ln.text)
            pages.append((page_no, lines, tables, float(page.height)))

    body = size_weights.most_common(1)[0][0] if size_weights else 0.0
    heading_sizes = sorted(
        {ln.size for _, lines, _, _ in pages for ln in lines if ln.bold and ln.size >= body * HEADING_RATIO},
        reverse=True,
    )

    for page_no, lines, tables, height in pages:
        if not lines and not tables:
            if ocr is None:
                doc.warnings.append(f"page {page_no}: no text layer and no OCR engine available")
                doc.degraded = True
                continue
            blocks, warnings, degraded = ocr.ocr_pdf_page(path, page_no)
            doc.blocks.extend(blocks)
            doc.warnings.extend(warnings)
            doc.degraded = doc.degraded or degraded
            continue
        doc.blocks.extend(_page_blocks(page_no, lines, tables, height, body, heading_sizes))
    return doc


def _page_blocks(page_no, lines, tables, height, body, heading_sizes) -> list[Block]:
    items: list[tuple[float, object]] = [(ln.top, ln) for ln in lines] + [(t.top, t) for t in tables]
    items.sort(key=lambda item: item[0])

    blocks: list[Block] = []
    para: list[str] = []
    prev: _Line | None = None

    def flush() -> None:
        if para and (text := clean_text(" ".join(para))):
            blocks.append(Block("text", text, page=page_no))
        para.clear()

    for _, item in items:
        if isinstance(item, _Table):
            flush()
            prev = None
            if not is_metadata_table(item.rows) and (b := table_block(item.rows, page_no)):
                blocks.append(b)
            continue
        ln = item
        in_margin = ln.top < height * CHROME_MARGIN or ln.bottom > height * (1 - CHROME_MARGIN)
        if ln.size < body * CHROME_RATIO and in_margin:
            continue  # running header/footer such as "Company | DOC-ID | Page 2"
        if ln.size in heading_sizes and ln.bold:
            flush()
            blocks.append(Block("heading", clean_text(ln.text), level=heading_sizes.index(ln.size) + 1, page=page_no))
            prev = None
            continue
        text = clean_text(ln.text)
        new_paragraph = (
            prev is None
            or text.startswith("- ")
            or ln.size != prev.size
            or ln.top - prev.bottom > ln.size * 0.6
        )
        if new_paragraph:
            flush()
        para.append(text)
        prev = ln
    flush()
    return blocks
