# Backend/RAG/bookshelf_fetch.py

import requests
from typing import Optional

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
TOOL = "MedFusionAI"
EMAIL = "student@example.com"


def fetch_bookshelf_definition(term: str) -> Optional[str]:
    """
    Fetch a clean, definition-focused passage from NCBI Bookshelf.
    Used ONLY for:
    - Disease definition
    - Basic pathophysiology
    """

    # -------------------------------
    # 1. Search Bookshelf
    # -------------------------------
    search_url = f"{NCBI_BASE}esearch.fcgi"
    search_params = {
        "db": "books",
        "term": f"{term}[Title]",
        "retmax": 1,
        "retmode": "json",
        "tool": TOOL,
        "email": EMAIL,
    }

    r = requests.get(search_url, params=search_params, timeout=20)
    r.raise_for_status()
    ids = r.json().get("esearchresult", {}).get("idlist", [])

    if not ids:
        return None

    # -------------------------------
    # 2. Fetch document summary
    # -------------------------------
    fetch_url = f"{NCBI_BASE}efetch.fcgi"
    fetch_params = {
        "db": "books",
        "id": ids[0],
        "retmode": "text",
        "rettype": "docsum",   # ✅ IMPORTANT FIX
        "tool": TOOL,
        "email": EMAIL,
    }

    r2 = requests.get(fetch_url, params=fetch_params, timeout=20)
    r2.raise_for_status()
    raw_text = r2.text.strip()

    if not raw_text:
        return None

    # -------------------------------
    # 3. Extract definition-like lines
    # -------------------------------
    lines = []
    for line in raw_text.splitlines():
        l = line.strip()
        if len(l) < 40:
            continue
        if any(
            bad in l.lower()
            for bad in ["treatment", "therapy", "trial", "drug", "dose"]
        ):
            continue
        lines.append(l)

    if not lines:
        return None

    # -------------------------------
    # 4. Return short, labeled context
    # -------------------------------
    summary = " ".join(lines[:3])

    return (
        "📘 NCBI Bookshelf (definition background):\n"
        + summary
    )
