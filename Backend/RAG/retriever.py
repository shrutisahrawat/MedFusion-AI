# Backend/RAG/retriever.py
"""
PubMed Retriever for MedFusion-AI (Python 3.9 Compatible)

Loads:
 - data/pubmed/index.faiss
 - data/pubmed/chunks.jsonl

Retrieves top-k scientific evidence and sends it to LLM for grounded answer.
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from Backend.llm.llama_client import generate_text_rag_answer


# ------------ Paths ------------
DATA_DIR = Path("data") / "pubmed"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
INDEX_PATH = DATA_DIR / "index.faiss"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class PubMedRetriever:
    def __init__(
        self,
        index_path=INDEX_PATH,
        chunks_path=CHUNKS_PATH,
        model_name=EMBED_MODEL,
    ):
        self.index_path = Path(index_path)
        self.chunks_path = Path(chunks_path)
        self.model_name = model_name

        self._index = None
        self._chunks = None
        self._embedder = None

    # -------------------------------
    # Load everything lazily
    # -------------------------------
    def _ensure_loaded(self):

        if self._embedder is None:
            self._embedder = SentenceTransformer(self.model_name)

        if self._index is None:
            if not self.index_path.exists():
                raise FileNotFoundError(
                    f"FAISS index missing at {self.index_path}. Run build_index.py first."
                )
            self._index = faiss.read_index(str(self.index_path))

        if self._chunks is None:
            if not self.chunks_path.exists():
                raise FileNotFoundError(
                    f"Chunks file missing at {self.chunks_path}. Run build_index.py first."
                )
            self._chunks = []
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        self._chunks.append(json.loads(line))
                    except:
                        continue

    def is_ready(self):
        try:
            self._ensure_loaded()
            return True
        except FileNotFoundError:
            return False

    # -------------------------------
    # RETRIEVE RAW TEXT CHUNKS
    # -------------------------------
    def retrieve(self, user_query, top_k=5):
        self._ensure_loaded()

        q_emb = self._embedder.encode([user_query], convert_to_numpy=True).astype("float32")

        if q_emb.ndim == 1:
            q_emb = np.expand_dims(q_emb, 0)

        distances, indices = self._index.search(q_emb, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self._chunks):
                results.append((self._chunks[idx]["text"], float(dist)))

        return results

    # -------------------------------
    # RETRIEVE FULL RECORDS
    # -------------------------------
    def retrieve_with_records(self, user_query, top_k=5):
        self._ensure_loaded()

        q_emb = self._embedder.encode([user_query], convert_to_numpy=True).astype("float32")

        if q_emb.ndim == 1:
            q_emb = np.expand_dims(q_emb, 0)

        distances, indices = self._index.search(q_emb, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self._chunks):
                rec = self._chunks[idx]
                results.append((rec, float(dist)))

        return results


# ---------- Singleton retriever ----------
_RETRIEVER_SINGLETON = None

def get_retriever(force_new=False):
    global _RETRIEVER_SINGLETON
    if _RETRIEVER_SINGLETON is None or force_new:
        _RETRIEVER_SINGLETON = PubMedRetriever()
    return _RETRIEVER_SINGLETON


# ---------- High-level Answer Function ----------
def answer_pubmed_question(user_question, top_k=5, return_contexts=False):
    retriever = get_retriever()

    if not retriever.is_ready():
        return "⚠️ PubMed index not found. Please run Backend/RAG/build_index.py."

    records = retriever.retrieve_with_records(user_question, top_k=top_k)
    context_texts = [rec["text"] for (rec, dist) in records]

    if not context_texts:
        out = "⚠️ No relevant PubMed passages found."
        return {"answer": out, "contexts": []} if return_contexts else out

    answer = generate_text_rag_answer(context_texts, user_question)

    if return_contexts:
        ctx = []
        for rec, dist in records:
            ctx.append({
                "text": rec.get("text"),
                "pmid": rec.get("pmid"),
                "chunk_id": rec.get("chunk_id"),
                "distance": dist,
            })
        return {"answer": answer, "contexts": ctx}

    return answer
