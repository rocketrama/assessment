"""The format-agnostic block model every loader emits (ARCHITECTURE §5.1).

A loader turns one file into a flat, ordered list of blocks: headings (with a
level, 1 = document title), text paragraphs and tables. The chunker rebuilds
the heading path from that order, so nothing downstream knows the file format.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

BlockKind = Literal["heading", "text", "table"]


@dataclass(frozen=True)
class Block:
    kind: BlockKind
    text: str  # heading text, paragraph text, or the table as markdown
    level: int = 0  # headings only: 1 = title, 2 = section, ...
    page: int | None = None  # 1-based; None for formats without pages
    rows: tuple[tuple[str, ...], ...] = ()  # tables only; first row is the header
    ocr_engine: str | None = None  # "vision" | "tesseract" when the text came from OCR
    low_confidence: bool = False  # OCR cross-check found a mismatch

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Block":
        rows = tuple(tuple(r) for r in data.get("rows", ()))
        return cls(**{**data, "rows": rows})


def clean_text(text: str) -> str:
    """Collapse whitespace; map the PDF bullet glyph (often U+FFFD or •) to '-'."""
    text = text.replace(" ", " ").replace("�", "•")
    text = " ".join(text.split())
    return "- " + text[1:].lstrip() if text.startswith("•") else text


def table_block(rows: list[list[str]], page: int | None = None, **extra: Any) -> Block | None:
    """A table block with cleaned cells, or None when the table is empty."""
    cleaned = [tuple(clean_text(c or "") for c in row) for row in rows]
    cleaned = [r for r in cleaned if any(r)]
    if not cleaned:
        return None
    width = max(len(r) for r in cleaned)
    cleaned = [r + ("",) * (width - len(r)) for r in cleaned]
    return Block("table", table_markdown(cleaned), page=page, rows=tuple(cleaned), **extra)


def table_markdown(rows: list[tuple[str, ...]] | tuple[tuple[str, ...], ...]) -> str:
    def line(cells: tuple[str, ...]) -> str:
        return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"

    header, *body = rows
    return "\n".join([line(header), "|" + "---|" * len(header), *(line(r) for r in body)])


def is_metadata_table(rows: list[list[str]] | tuple[tuple[str, ...], ...]) -> bool:
    """The key/value "Document ID / Version / Effective date ..." box at the top of
    each document. It repeats manifest data (the manifest is authoritative, §2.2),
    so it is not indexed. Markdown versions put a "Field | Value" header row first."""
    return any(row and (row[0] or "").strip().lower().startswith("document id") for row in rows[:2])


@dataclass
class ParsedDocument:
    """Loader output for one file."""

    blocks: list[Block] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # OCR failed or fell back: don't cache, so the next startup tries again.
    degraded: bool = False
