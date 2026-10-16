"""DOCX loader (python-docx): Title / Heading N styles, paragraphs, lists and tables in
body order. Headers and footers are page chrome and are not read."""

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.ingestion.blocks import Block, ParsedDocument, clean_text, is_metadata_table, table_block

_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _heading_level(style_name: str) -> int | None:
    if style_name == "Title":
        return 1
    if style_name.startswith("Heading "):
        try:
            return int(style_name.split()[1]) + 1  # Heading 1 sits under the title
        except (IndexError, ValueError):
            return 2
    return None


def _is_list(paragraph: Paragraph, style_name: str) -> bool:
    ppr = paragraph._p.pPr
    return style_name.startswith("List") or (ppr is not None and ppr.numPr is not None)


def load_docx(path: str) -> ParsedDocument:
    document = docx.Document(path)
    blocks: list[Block] = []
    for element in document.element.body.iterchildren():
        tag = element.tag.removeprefix(_W_NS)
        if tag == "p":
            paragraph = Paragraph(element, document)
            text = clean_text(paragraph.text)
            if not text:
                continue
            style = paragraph.style.name if paragraph.style is not None else ""
            if (level := _heading_level(style)) is not None:
                blocks.append(Block("heading", text, level=level))
            elif _is_list(paragraph, style):
                blocks.append(Block("text", text if text.startswith("- ") else f"- {text}"))
            else:
                blocks.append(Block("text", text))
        elif tag == "tbl":
            rows = [[cell.text for cell in row.cells] for row in Table(element, document).rows]
            if not is_metadata_table(rows) and (b := table_block(rows)):
                blocks.append(b)
    return ParsedDocument(blocks)
