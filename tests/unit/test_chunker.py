"""Section chunking, locators and token limits (ARCHITECTURE §5.4, TASK.pdf §4.2)."""

from app.ingestion.blocks import Block, table_block
from app.ingestion.chunker import MAX_TOKENS, chunk_document
from app.tokens import count_tokens


def H(text, level, page=None):
    return Block("heading", text, level=level, page=page)


def T(text, page=None):
    return Block("text", text, page=page)


def test_one_chunk_per_section_with_heading_prefix_and_locator():
    blocks = [
        H("Policy", 1),
        T("Intro text."),
        H("1. Sick leave", 2),
        T("Ten days."),
        T("Certificate after 3 days."),
        H("2. Parental leave", 2),
        H("2.1 Primary", 3),
        T("Twenty weeks."),
    ]
    chunks = chunk_document("DOC", blocks)
    assert [(c.chunk_id, c.locator, c.heading_path) for c in chunks] == [
        ("DOC::000", "Policy", ("Policy",)),
        ("DOC::001", "1. Sick leave", ("Policy", "1. Sick leave")),
        ("DOC::002", "2.1 Primary", ("Policy", "2. Parental leave", "2.1 Primary")),
    ]
    assert chunks[1].text == "Policy > 1. Sick leave\n\nTen days.\n\nCertificate after 3 days."
    assert chunks[1].tokens == count_tokens(chunks[1].text)


def test_sibling_heading_pops_the_stack():
    chunks = chunk_document("DOC", [H("P", 1), H("A", 2), T("a"), H("B", 2), T("b"), H("Q", 1), T("q")])
    assert [c.heading_path for c in chunks] == [("P", "A"), ("P", "B"), ("Q",)]


def test_headless_document_is_chunked_per_page_with_page_locators():
    chunks = chunk_document("SCAN", [T("first", page=1), T("more", page=1), T("second", page=2)])
    assert [(c.locator, c.text) for c in chunks] == [("p. 1", "first\n\nmore"), ("p. 2", "second")]


def test_long_section_splits_under_max_and_keeps_locator():
    sentence = "Employees accrue leave according to the schedule in this section. "
    blocks = [H("Policy", 1), H("5. Accrual", 2)] + [T(sentence * 12) for _ in range(6)]
    chunks = chunk_document("DOC", blocks)
    assert len(chunks) > 1
    assert all(c.tokens <= MAX_TOKENS for c in chunks)
    assert {c.locator for c in chunks} == {"5. Accrual"}


def test_single_oversized_paragraph_is_split_on_sentences():
    text = " ".join(f"Rule number {i} applies to every office." for i in range(200))
    chunks = chunk_document("DOC", [H("P", 1), T(text)])
    assert len(chunks) > 1 and all(c.tokens <= MAX_TOKENS for c in chunks)
    assert " ".join(c.text.split("\n\n", 1)[1] for c in chunks) == text


def test_big_table_splits_by_rows_with_header_repeated():
    rows = [["Date", "Holiday"]] + [[f"2026-01-{d:02d}", f"Holiday name number {d} " * 4] for d in range(1, 29)] * 2
    chunks = chunk_document("CAL", [H("Calendar", 1), table_block(rows)])
    assert len(chunks) > 1
    for c in chunks:
        assert c.has_table and c.tokens <= MAX_TOKENS
        body = c.text.split("\n\n", 1)[1]
        assert body.startswith("| Date | Holiday |\n|---|---|\n")
    body_rows = sum(c.text.count("\n| 2026-") for c in chunks)
    assert body_rows == len(rows) - 1  # no row lost or duplicated


def test_ocr_metadata_is_carried():
    blocks = [Block("text", "scanned", page=1, ocr_engine="tesseract", low_confidence=True)]
    (chunk,) = chunk_document("SCAN", blocks)
    assert (chunk.ocr_engine, chunk.low_confidence, chunk.locator) == ("tesseract", True, "p. 1")


def test_empty_document_has_no_chunks():
    assert chunk_document("DOC", [H("Only a title", 1)]) == []
