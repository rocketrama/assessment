"""Section-based chunking (ARCHITECTURE §5.4).

Chunks follow sections, so each chunk's locator is its nearest heading (§4.2) and
a rule stays with its conditions. Documents without headings are chunked per
page and cited as `p. N`. Target ~350 tokens, hard max 500, counted with
cl100k_base including the heading-path prefix (the prefix is corpus text too).
Tables are never cut mid-row; a table over the limit is split by row groups
with the header row repeated.
"""

import re
from dataclasses import dataclass

from app.ingestion.blocks import Block, table_markdown
from app.tokens import count_tokens

TARGET_TOKENS = 350
MAX_TOKENS = 500

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    ordinal: int
    heading_path: tuple[str, ...]
    locator: str
    page: int | None
    text: str  # heading-path prefix + body; exactly what an LLM would see
    tokens: int
    has_table: bool
    ocr_engine: str | None
    low_confidence: bool
    suspicious: bool = False


@dataclass
class _Section:
    path: tuple[str, ...]
    blocks: list[Block]


def _sections(blocks: list[Block]) -> list[_Section]:
    has_headings = any(b.kind == "heading" for b in blocks)
    stack: list[tuple[int, str]] = []
    sections: list[_Section] = []
    for b in blocks:
        if b.kind == "heading":
            while stack and stack[-1][0] >= b.level:
                stack.pop()
            stack.append((b.level, b.text))
            continue
        path = tuple(text for _, text in stack)
        last = sections[-1] if sections else None
        same_section = last is not None and last.path == path
        if same_section and not has_headings:
            same_section = last.blocks[-1].page == b.page  # headless docs: one section per page
        if same_section:
            last.blocks.append(b)
        else:
            sections.append(_Section(path, [b]))
    return sections


def _split_long_text(text: str, budget: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_RE.split(text):
        candidate = f"{current} {sentence}".strip()
        if current and count_tokens(candidate) > budget:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)
    # A single sentence longer than the budget: split on words.
    out: list[str] = []
    for piece in pieces:
        if count_tokens(piece) <= budget:
            out.append(piece)
            continue
        words, part = piece.split(), ""
        for word in words:
            candidate = f"{part} {word}".strip()
            if part and count_tokens(candidate) > budget:
                out.append(part)
                part = word
            else:
                part = candidate
        if part:
            out.append(part)
    return out


def _split_table(block: Block, budget: int) -> list[str]:
    header, *body = block.rows
    groups: list[str] = []
    rows: list[tuple[str, ...]] = []
    for row in body:
        if rows and count_tokens(table_markdown([header, *rows, row])) > budget:
            groups.append(table_markdown([header, *rows]))
            rows = []
        rows.append(row)
    if rows or not groups:
        groups.append(table_markdown([header, *rows]))
    return groups


def _pieces(block: Block, budget: int) -> list[tuple[str, bool]]:
    """(text, is_table) units that each fit in `budget` tokens."""
    if count_tokens(block.text) <= budget:
        return [(block.text, block.kind == "table")]
    if block.kind == "table" and len(block.rows) > 1:
        return [(t, True) for t in _split_table(block, budget)]
    return [(t, False) for t in _split_long_text(block.text, budget)]


def chunk_document(doc_id: str, blocks: list[Block]) -> list[Chunk]:
    chunks: list[Chunk] = []

    for section in _sections(blocks):
        prefix = " > ".join(section.path)
        prefix_text = f"{prefix}\n\n" if prefix else ""
        budget = MAX_TOKENS - count_tokens(prefix_text)
        locator_page = section.blocks[0].page
        locator = section.path[-1] if section.path else f"p. {locator_page or 1}"

        current: list[str] = []
        current_tokens = 0
        has_table = False
        meta_blocks: list[Block] = []

        def emit() -> None:
            nonlocal current, current_tokens, has_table, meta_blocks
            if not current:
                return
            text = prefix_text + "\n\n".join(current)
            engines = [b.ocr_engine for b in meta_blocks if b.ocr_engine]
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::{len(chunks):03d}",
                    doc_id=doc_id,
                    ordinal=len(chunks),
                    heading_path=section.path,
                    locator=locator,
                    page=meta_blocks[0].page if meta_blocks else locator_page,
                    text=text,
                    tokens=count_tokens(text),
                    has_table=has_table,
                    ocr_engine=engines[0] if engines else None,
                    low_confidence=any(b.low_confidence for b in meta_blocks),
                )
            )
            current, current_tokens, has_table, meta_blocks = [], 0, False, []

        for block in section.blocks:
            for text, is_table in _pieces(block, budget):
                piece_tokens = count_tokens(text) + 1  # + the joining blank line
                if current and (current_tokens + piece_tokens > budget or current_tokens >= TARGET_TOKENS):
                    emit()
                current.append(text)
                current_tokens += piece_tokens
                has_table = has_table or is_table
                meta_blocks.append(block)
        emit()

    return chunks
