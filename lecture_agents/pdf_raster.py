from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF


def rasterize_pdf_to_pngs(pdf_path: Path, out_dir: Path, zoom: float = 2.0) -> list[Path]:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    matrix = fitz.Matrix(zoom, zoom)
    paths: list[Path] = []
    try:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            name = f"slide_{i + 1:03d}.png"
            dest = out_dir / name
            pix.save(dest.as_posix())
            paths.append(dest)
    finally:
        doc.close()
    return paths
