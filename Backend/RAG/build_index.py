# backend/rag/build_index.py
"""
Build pipeline:
 - fetch PubMed PMIDs for a term
 - fetch titles+abstracts
 - write data/pubmed/raw_pubmed.jsonl
 - chunk text -> write data/pubmed/chunks.jsonl
 - embed chunks -> build faiss -> save to data/pubmed/index.faiss
"""

import json
import os
from pathlib import Path
from typing import List

import faiss
import numpy as np

from Backend.RAG.pubmed_fetch import (
    fetch_pubmed_ids,
    fetch_pubmed_article,
    clean_text,
    chunk_text,
    embed_text_chunks,
    create_faiss_index,
)


# Output files under project root data/pubmed/
DATA_DIR = Path("data") / "pubmed"
RAW_PATH = DATA_DIR / "raw_pubmed.jsonl"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
INDEX_PATH = DATA_DIR / "index.faiss"


def build_pubmed_index(term: str = "pneumonia chest x ray", max_results: int = 150, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[build_index] Searching PubMed for: '{term}' (max {max_results})")
    pmids = fetch_pubmed_ids(term, max_results=max_results)
    print(f"[build_index] {len(pmids)} PMIDs found")

    all_chunk_texts: List[str] = []
    chunk_records = []  # for writing chunks.jsonl

    # write raw articles and chunk records
    with open(RAW_PATH, "w", encoding="utf-8") as raw_f, open(CHUNKS_PATH, "w", encoding="utf-8") as chunk_f:
        for pmid in pmids:
            try:
                art = fetch_pubmed_article(pmid)
            except Exception as e:
                print(f"[build_index] Failed fetching PMID {pmid}: {e}")
                continue

            # write raw article
            raw_f.write(json.dumps(art, ensure_ascii=False) + "\n")

            full_text = (art.get("title") or "") + ". " + (art.get("abstract") or "")
            cleaned = clean_text(full_text)
            chunks = chunk_text(cleaned, max_tokens=384)
            for i, c in enumerate(chunks):
                rec = {"pmid": pmid, "chunk_id": i, "text": c}
                chunk_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                all_chunk_texts.append(c)
                chunk_records.append(rec)

    if len(all_chunk_texts) == 0:
        print("[build_index] No chunks were produced. Exiting.")
        return

    print(f"[build_index] Total chunks: {len(all_chunk_texts)}. Computing embeddings (this may take time)...")
    embeddings = embed_text_chunks(all_chunk_texts, model_name=embed_model)
    print(f"[build_index] Embeddings shape: {embeddings.shape}")

    print("[build_index] Building FAISS index...")
    index = create_faiss_index(embeddings)
    faiss.write_index(index, str(INDEX_PATH))
    print(f"[build_index] FAISS index saved to: {INDEX_PATH}")

    print(f"[build_index] Done. Raw: {RAW_PATH}, Chunks: {CHUNKS_PATH}, Index: {INDEX_PATH}")


if __name__ == "__main__":
    print("Run this file to manually build the PubMed index if needed.")
