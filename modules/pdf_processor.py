"""PDF resume text extraction using PyMuPDF."""

from __future__ import annotations

from pathlib import Path


def extract_pdf_text(file_path: str | Path) -> dict:
    """Extract text from PDF resume."""
    import fitz  # PyMuPDF

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    text_parts = []
    page_count = 0
    try:
        doc = fitz.open(str(path))
        page_count = len(doc)
        for page in doc:
            text_parts.append(page.get_text("text"))
        doc.close()
    except Exception as exc:
        raise RuntimeError(f"PDF extraction failed: {exc}") from exc

    full_text = "\n".join(t.strip() for t in text_parts if t.strip())
    return {
        "text": full_text,
        "metadata": {
            "source": path.name,
            "type": "resume_pdf",
            "pages": page_count,
            "char_count": len(full_text),
        },
    }
