"""Markdown loader: `#` headings, paragraphs, lists and pipe tables (markdown-it-py).

Also parses the markdown that OCR returns for scanned pages.
"""

from markdown_it import MarkdownIt
from markdown_it.token import Token

from app.ingestion.blocks import Block, ParsedDocument, clean_text, is_metadata_table, table_block

_MD = MarkdownIt("commonmark").enable("table")


def _inline_text(token: Token) -> str:
    parts = []
    for child in token.children or []:
        if child.type in ("text", "code_inline"):
            parts.append(child.content)
        elif child.type in ("softbreak", "hardbreak"):
            parts.append(" ")
    return clean_text("".join(parts))


def parse_markdown_text(text: str, page: int | None = None) -> list[Block]:
    blocks: list[Block] = []
    heading_level: int | None = None
    list_depth = 0
    table_rows: list[list[str]] | None = None
    row: list[str] = []

    for tok in _MD.parse(text):
        t = tok.type
        if t == "heading_open":
            heading_level = int(tok.tag[1])
        elif t in ("bullet_list_open", "ordered_list_open"):
            list_depth += 1
        elif t in ("bullet_list_close", "ordered_list_close"):
            list_depth -= 1
        elif t == "table_open":
            table_rows = []
        elif t == "tr_open":
            row = []
        elif t == "tr_close" and table_rows is not None:
            table_rows.append(row)
        elif t == "table_close" and table_rows is not None:
            if not is_metadata_table(table_rows) and (b := table_block(table_rows, page)):
                blocks.append(b)
            table_rows = None
        elif t in ("fence", "code_block"):
            if content := tok.content.strip():
                blocks.append(Block("text", content, page=page))
        elif t == "inline":
            content = _inline_text(tok)
            if table_rows is not None:
                row.append(content)
            elif heading_level is not None:
                if content:
                    blocks.append(Block("heading", content, level=heading_level, page=page))
                heading_level = None
            elif content:
                blocks.append(Block("text", f"- {content}" if list_depth else content, page=page))
    return blocks


def load_markdown(path: str) -> ParsedDocument:
    with open(path, encoding="utf-8-sig") as f:
        return ParsedDocument(parse_markdown_text(f.read()))
