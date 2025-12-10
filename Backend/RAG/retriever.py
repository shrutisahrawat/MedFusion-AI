# backend/rag/retriever.py

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class PubMedRetriever:
    def __init__(self, index_path=None, chunks=None, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        index_path: path to saved FAISS index (.index file)
        chunks: list of text chunks corresponding to FAISS vectors
        """
        self.model = SentenceTransformer(model_name)
        self.chunks = chunks or []

        if index_path:
            self.index = faiss.read_index(index_path)
        else:
            self.index = None

    def set_index(self, index, chunks):
        """Set FAISS index and chunks manually"""
        self.index = index
        self.chunks = chunks

    def save_index(self, index_path):
        """Save FAISS index to disk"""
        if self.index:
            faiss.write_index(self.index, index_path)

    def retrieve(self, query, top_k=5):
        """
        Retrieve top_k most relevant chunks for the query.
        Returns a list of (chunk, distance) tuples.
        """
        if not self.index or not self.chunks:
            raise ValueError("FAISS index or chunks not loaded.")

        # Embed the query
        query_vec = self.model.encode([query], convert_to_numpy=True)

        # Search
        distances, indices = self.index.search(query_vec, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], dist))
        return results

# Example usage
if __name__ == "__main__":
    # Suppose you already have chunks and FAISS index from pubmed_fetch.py
    from backend.rag.pubmed_fetch import embed_text_chunks, create_faiss_index, chunk_text, clean_text, fetch_pubmed_ids, fetch_pubmed_article

    term = "COVID-19 vaccine"
    pmids = fetch_pubmed_ids(term, max_results=3)
    all_chunks = []

    for pmid in pmids:
        article = fetch_pubmed_article(pmid)
        text = article["title"] + ". " + article["abstract"]
        clean = clean_text(text)
        chunks = chunk_text(clean, max_tokens=384)
        all_chunks.extend(chunks)

    embeddings = embed_text_chunks(all_chunks)
    index = create_faiss_index(embeddings)

    retriever = PubMedRetriever()
    retriever.set_index(index, all_chunks)

    query = "Effectiveness of COVID-19 vaccines"
    results = retriever.retrieve(query, top_k=3)
    for i, (chunk, dist) in enumerate(results):
        print(f"\nResult {i+1} (distance={dist:.4f}): {chunk}")
