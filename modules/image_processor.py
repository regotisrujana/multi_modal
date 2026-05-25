"""Image and certificate OCR using EasyOCR."""

from __future__ import annotations

from pathlib import Path

_ocr_reader = None


def _get_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr

        # English; add ['en'] only for faster init on student machines
        _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _ocr_reader


def extract_image_text(file_path: str | Path) -> dict:
    """Run OCR on images (resumes scans, LinkedIn screenshots, certificates)."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        # OpenCV preprocessing improves OCR on screenshots/certificates
        import cv2

        img = cv2.imread(str(path))
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            ocr_path = str(path.parent / f"_ocr_{path.name}")
            cv2.imwrite(ocr_path, gray)
            ocr_input = ocr_path
        else:
            ocr_input = str(path)

        reader = _get_reader()
        results = reader.readtext(ocr_input, detail=0, paragraph=True)
        if img is not None and Path(ocr_path).exists():
            Path(ocr_path).unlink(missing_ok=True)
        full_text = "\n".join(results) if results else ""
    except Exception as exc:
        raise RuntimeError(f"OCR failed for {path.name}: {exc}") from exc

    doc_type = "linkedin_screenshot" if "linkedin" in path.name.lower() else "image"
    if "cert" in path.name.lower():
        doc_type = "certificate"

    return {
        "text": full_text,
        "metadata": {
            "source": path.name,
            "type": doc_type,
            "char_count": len(full_text),
        },
    }
