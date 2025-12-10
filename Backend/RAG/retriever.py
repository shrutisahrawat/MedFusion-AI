# backend/rag/retriever.py

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle

class PubMedRetriever:
    def __init__(self, index_path=None, chunks=None, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chunks = chunks or []

        if index_path:
            self.index = faiss.read_index(index_path)
        else:
            self.index = None

    def set_index(self, index, chunks):
        self.index = index
        self.chunks = chunks

    def save_index(self, index_path):
        if self.index:
            faiss.write_index(self.index, index_path)

    def retrieve(self, query, top_k=5):
        if not self.index or not self.chunks:
            raise ValueError("FAISS index or chunks not loaded.")

        query_vec = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], dist))
        return results

if __name__ == "__main__":
    # Load index and chunks saved by pubmed_fetch.py
    with open("index_data/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    index = faiss.read_index("index_data/pubmed.index")

    retriever = PubMedRetriever()
    retriever.set_index(index, chunks)

    query = "Effectiveness of COVID-19 vaccines"
    results = retriever.retrieve(query, top_k=3)
    for i, (chunk, dist) in enumerate(results):
        print(f"\nResult {i+1} (distance={dist:.4f}): {chunk}")
