"""HTML loader (beautifulsoup4): h1–h6, paragraphs, lists and tables in document order.

Page chrome (nav, header, footer) and non-content tags are dropped before walking.
"""

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from app.ingestion.blocks import Block, ParsedDocument, clean_text, is_metadata_table, table_block

_DROP = ["script", "style", "nav", "header", "footer", "noscript", "template", "iframe", "form"]
_HEADINGS = {f"h{i}": i for i in range(1, 7)}
_TEXT_TAGS = {"p", "blockquote", "pre", "dt", "dd", "figcaption", "caption", "address"}


def _walk(element: Tag, blocks: list[Block]) -> None:
    for child in element.children:
        if isinstance(child, Comment):
            continue
        if isinstance(child, NavigableString):
            if text := clean_text(str(child)):
                blocks.append(Block("text", text))
            continue
        if not isinstance(child, Tag):
            continue
        name = child.name
        if name in _HEADINGS:
            if text := clean_text(child.get_text(" ")):
                blocks.append(Block("heading", text, level=_HEADINGS[name]))
        elif name == "table":
            rows = [
                [cell.get_text(" ") for cell in tr.find_all(["th", "td"])]
                for tr in child.find_all("tr")
            ]
            if not is_metadata_table(rows) and (b := table_block(rows)):
                blocks.append(b)
        elif name in ("ul", "ol"):
            for li in child.find_all("li"):
                # Nested lists are flattened; each item once, without its sub-items' text.
                own = "".join(
                    s if isinstance(s, str) else s.get_text(" ")
                    for s in li.contents
                    if not (isinstance(s, Tag) and s.name in ("ul", "ol"))
                )
                if text := clean_text(own):
                    blocks.append(Block("text", f"- {text}"))
        elif name in _TEXT_TAGS:
            if text := clean_text(child.get_text(" ")):
                blocks.append(Block("text", text))
        else:
            _walk(child, blocks)


def load_html(path: str) -> ParsedDocument:
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        soup = BeautifulSoup(f.read(), "lxml")
    for tag in soup(_DROP):
        tag.decompose()
    blocks: list[Block] = []
    _walk(soup.body or soup, blocks)
    return ParsedDocument(blocks)
