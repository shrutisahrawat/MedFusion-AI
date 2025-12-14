# Backend/pdf/extract_text.py

from __future__ import annotations
from typing import Union
from pathlib import Path

from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: Union[str, Path]) -> str:
    """
    Extracts text from a PDF using pypdf. Works for text-based PDFs.
    If the PDF is scanned image-only, this will return little/no text (OCR needed).
    """
    pdf_path = Path(pdf_path)
    reader = PdfReader(str(pdf_path))

    parts = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if txt.strip():
            parts.append(f"\n--- Page {i+1} ---\n{txt}")

    return "\n".join(parts).strip()
