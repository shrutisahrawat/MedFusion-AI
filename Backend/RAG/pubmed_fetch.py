# backend/rag/pubmed_fetch.py
"""
PubMed fetch + clean + chunk + embed helpers (no disk IO here).
"""

import re
from typing import List, Dict
from xml.etree import ElementTree as ET

import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


# -------------------------
# PubMed fetching helpers
# -------------------------
def fetch_pubmed_ids(term: str, max_results: int = 50) -> List[str]:
    """Search PubMed and return PMIDs for the query."""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": term,
        "retmax": max_results,
        "retmode": "xml",
    }
    resp = requests.get(base_url, params=params, timeout=30)
    resp.raise_for_status()
    tree = ET.fromstring(resp.content)
    ids = [elem.text for elem in tree.findall(".//Id")]
    return ids


def fetch_pubmed_article(pmid: str) -> Dict[str, str]:
    """Fetch a single article (title + abstract) for a PMID."""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": pmid, "retmode": "xml"}
    resp = requests.get(base_url, params=params, timeout=30)
    resp.raise_for_status()
    tree = ET.fromstring(resp.content)

    title_elem = tree.find(".//ArticleTitle")
    title = title_elem.text if title_elem is not None and title_elem.text else ""

    abstract_parts = []
    for abstr in tree.findall(".//AbstractText"):
        if abstr.text:
            abstract_parts.append(abstr.text)
    abstract = " ".join(abstract_parts)

    return {"pmid": pmid, "title": title, "abstract": abstract}


# -------------------------
# Text cleaning & chunking
# -------------------------
def clean_text(text: str) -> str:
    """Strip tags and excessive whitespace."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, max_tokens: int = 384) -> List[str]:
    """
    Naive sentence based chunking approximating tokens by words.
    Returns list of chunk strings.
    """
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?]) +", text)
    chunks = []
    cur = ""
    for s in sentences:
        if len((cur + " " + s).split()) <= max_tokens:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = s.strip()
    if cur:
        chunks.append(cur)
    return chunks


# -------------------------
# Embedding helpers
# -------------------------
def embed_text_chunks(chunks: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Return numpy float32 embeddings for the list of chunk strings."""
    if not chunks:
        return np.zeros((0, 384), dtype="float32")  # safe empty shape
    model = SentenceTransformer(model_name)
    emb = model.encode(chunks, convert_to_numpy=True)
    return emb.astype("float32")


def create_faiss_index(embeddings: np.ndarray):
    """Create a simple L2 FAISS index and add embeddings."""
    if embeddings is None or embeddings.shape[0] == 0:
        raise ValueError("No embeddings to index.")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index
