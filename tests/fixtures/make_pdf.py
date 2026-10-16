"""Hand-built PDFs for loader tests: standard Type1 fonts, ruled tables, a blank
(scanned-like) page. No PDF library needed, so tests don't depend on ./data."""

from dataclasses import dataclass, field

PAGE_W, PAGE_H = 595, 842


@dataclass
class PdfPage:
    ops: list[str] = field(default_factory=list)

    def text(self, x: float, y: float, s: str, size: float = 10, bold: bool = False) -> "PdfPage":
        s = s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        self.ops.append(f"BT /{'F2' if bold else 'F1'} {size} Tf {x} {y} Td ({s}) Tj ET")
        return self

    def table(self, x: float, y_top: float, col_widths: list[float], rows: list[list[str]], row_h: float = 18) -> "PdfPage":
        xs = [x]
        for w in col_widths:
            xs.append(xs[-1] + w)
        ys = [y_top - i * row_h for i in range(len(rows) + 1)]
        self.ops.append("0.5 w")
        for yy in ys:
            self.ops.append(f"{xs[0]} {yy} m {xs[-1]} {yy} l S")
        for xx in xs:
            self.ops.append(f"{xx} {ys[0]} m {xx} {ys[-1]} l S")
        for r, row in enumerate(rows):
            for c, cell in enumerate(row):
                self.text(xs[c] + 4, ys[r] - 13, cell, size=9, bold=(r == 0))
        return self


def build_pdf(pages: list[PdfPage]) -> bytes:
    n = len(pages)
    # 1 catalog, 2 pages, 3 F1, 4 F2, then (page, content) pairs
    page_ids = [5 + 2 * i for i in range(n)]
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{' '.join(f'{p} 0 R' for p in page_ids)}] /Count {n} >>".encode(),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        4: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
    }
    for pid, page in zip(page_ids, pages):
        stream = "\n".join(page.ops).encode("latin-1")
        objects[pid] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {pid + 1} 0 R >>"
        ).encode()
        objects[pid + 1] = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"

    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for oid in sorted(objects):
        offsets[oid] = len(out)
        out += f"{oid} 0 obj\n".encode() + objects[oid] + b"\nendobj\n"
    xref = len(out)
    size = max(objects) + 1
    out += f"xref\n0 {size}\n0000000000 65535 f \n".encode()
    for oid in range(1, size):
        out += f"{offsets[oid]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def sample_policy_pdf(with_scanned_page: bool = False) -> bytes:
    """Title, two sections, a two-line paragraph, a ruled table, a metadata table and
    a footer — the same layout as the corpus PDFs, with invented content."""
    p1 = PdfPage()
    p1.text(50, 790, "Test Policy", size=18, bold=True)
    p1.table(50, 770, [120, 200], [["Document ID", "TEST-POL"], ["Version", "1.0"]])
    p1.text(50, 715, "1. Leave rules", size=13, bold=True)
    p1.text(50, 695, "Staff get 7 test days each year. This line")
    p1.text(50, 682, "continues the same paragraph.")
    p1.text(50, 655, "A second paragraph starts here.")
    p1.text(50, 625, "2. Rates", size=13, bold=True)
    p1.table(50, 605, [120, 120], [["Band", "Days"], ["A", "1.5"], ["B", "2.0"]])
    p1.text(50, 20, "Test Co | TEST-POL | Page 1", size=7)
    pages = [p1]
    if with_scanned_page:
        pages.append(PdfPage())  # no text layer
    return build_pdf(pages)
