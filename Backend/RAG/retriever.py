# Backend/RAG/retriever.py

import json
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from Backend.RAG.pubmed_fetch import (
    fetch_pubmed_ids,
    fetch_pubmed_article,
    clean_text,
    chunk_text,
)
from Backend.RAG.bookshelf_fetch import fetch_bookshelf_definition


# ======================================================
# Paths
# ======================================================
DATA_DIR = Path("data") / "pubmed"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
INDEX_PATH = DATA_DIR / "index.faiss"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ======================================================
# 🔥 Query intent detection
# ======================================================
def is_definition_query(question: str) -> bool:
    keywords = [
        "what is",
        "define",
        "definition",
        "meaning of",
        "explain",
    ]
    q = question.lower()
    return any(k in q for k in keywords)


def is_mechanism_query(question: str) -> bool:
    keywords = [
        "how does",
        "pathophysiology",
        "mechanism",
        "what happens in",
    ]
    q = question.lower()
    return any(k in q for k in keywords)


# ======================================================
# Retriever
# ======================================================
class PubMedRetriever:
    def __init__(self):
        self._model = SentenceTransformer(EMBED_MODEL)
        self._index = faiss.read_index(str(INDEX_PATH))
        self._chunks: List[Dict] = []

        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    self._chunks.append(json.loads(line))
                except:
                    continue

    # --------------------------------------------------
    # Compatibility method (DO NOT DELETE)
    # --------------------------------------------------
    def is_ready(self) -> bool:
        return self._index is not None and len(self._chunks) > 0

    # --------------------------------------------------
    # Local FAISS search
    # --------------------------------------------------
    def _search(self, query: str, top_k: int):
        q_emb = self._model.encode([query], convert_to_numpy=True).astype("float32")
        D, I = self._index.search(q_emb, top_k)

        results = []
        for dist, idx in zip(D[0], I[0]):
            if 0 <= idx < len(self._chunks):
                results.append((self._chunks[idx], float(dist)))
        return results

    # --------------------------------------------------
    # Evidence quality check
    # --------------------------------------------------
    def _needs_refresh(self, results, threshold=0.35):
        if not results:
            return True
        sims = [1 - d for _, d in results]
        return sum(sims) / len(sims) < threshold

    # --------------------------------------------------
    # Fetch + store new PubMed
    # --------------------------------------------------
    def _fetch_and_store(self, query: str, max_results=3):
        pmids = fetch_pubmed_ids(query, max_results)
        new_records = []

        for pmid in pmids:
            art = fetch_pubmed_article(pmid)
            text = clean_text(f"{art['title']}. {art['abstract']}")
            for i, ch in enumerate(chunk_text(text, max_tokens=384)):
                new_records.append({
                    "pmid": pmid,
                    "text": ch,
                })

        if not new_records:
            return

        vecs = self._model.encode(
            [r["text"] for r in new_records],
            convert_to_numpy=True
        ).astype("float32")

        self._index.add(vecs)
        faiss.write_index(self._index, str(INDEX_PATH))

        with open(CHUNKS_PATH, "a", encoding="utf-8") as f:
            for r in new_records:
                f.write(json.dumps(r) + "\n")
                self._chunks.append(r)

    # --------------------------------------------------
    # 🔥 MAIN retrieval (IMPROVED)
    # --------------------------------------------------
    def retrieve(self, query: str, top_k: int = 6):
        results = self._search(query, top_k)

        if self._needs_refresh(results):
            self._fetch_and_store(query)
            results = self._search(query, top_k)

        # 🔥 semantic categorization
        definition_support = []
        mechanism_support = []
        research_support = []

        for rec, _ in results:
            txt = rec["text"].lower()

            if any(k in txt for k in ["is a", "refers to", "defined as"]):
                definition_support.append(rec)

            elif any(k in txt for k in ["pathophysiology", "mechanism", "airway inflammation"]):
                mechanism_support.append(rec)

            else:
                research_support.append(rec)

        return {
            "definition_support": definition_support[:2],
            "mechanism_support": mechanism_support[:2],
            "research_support": research_support[:2],
        }


# ======================================================
# Public API (DO NOT DELETE)
# ======================================================
_retriever = PubMedRetriever()


def answer_pubmed_question(user_question: str, top_k: int = 5):
    """
    High-level API used by legacy components.
    Now returns structured evidence + optional Bookshelf.
    """

    if not _retriever.is_ready():
        return "⚠️ PubMed index not found. Please build index first."

    buckets = _retriever.retrieve(user_question, top_k)

    # 🔥 Bookshelf only for definitions
    bookshelf_text = None
    if is_definition_query(user_question):
        bookshelf_text = fetch_bookshelf_definition(user_question)

    # Flatten evidence for LLM compatibility
    context_records = (
        buckets["definition_support"] +
        buckets["mechanism_support"] +
        buckets["research_support"]
    )

    pmids = list(dict.fromkeys([r["pmid"] for r in context_records]))

    return {
        "contexts": context_records,
        "pmids": pmids,
        "bookshelf": bookshelf_text,
    }
