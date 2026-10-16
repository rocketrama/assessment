"""Pick a loader by file extension; every loader returns the same block model."""

import os

from app.ingestion.blocks import ParsedDocument
from app.ingestion.loaders.docx import load_docx
from app.ingestion.loaders.html import load_html
from app.ingestion.loaders.markdown import load_markdown
from app.ingestion.loaders.ocr import PageOcr
from app.ingestion.loaders.pdf import load_pdf


def load_document(path: str, ocr: PageOcr | None = None) -> ParsedDocument:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf(path, ocr)
    if ext == ".docx":
        return load_docx(path)
    if ext in (".html", ".htm"):
        return load_html(path)
    if ext == ".md":
        return load_markdown(path)
    raise ValueError(f"unsupported file type: {ext}")
