# backend/rag/build_index.py

from Backend.RAG.pubmed_fetch  import fetch_pubmed_ids, fetch_pubmed_article, clean_text, chunk_text, embed_text_chunks, create_faiss_index
import os
import pickle
import faiss

def build_and_save_index(term="COVID-19 vaccine", max_results=5):
    print(f"Building index for term '{term}' with max results {max_results}")
    pmids = fetch_pubmed_ids(term, max_results=max_results)

    all_chunks = []
    for pmid in pmids:
        article = fetch_pubmed_article(pmid)
        text = article["title"] + ". " + article["abstract"]
        clean = clean_text(text)
        chunks = chunk_text(clean, max_tokens=384)
        all_chunks.extend(chunks)

    embeddings = embed_text_chunks(all_chunks)
    index = create_faiss_index(embeddings)

    os.makedirs("index_data", exist_ok=True)
    faiss.write_index(index, "index_data/pubmed.index")
    with open("index_data/chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    print("Index built and saved to 'index_data/'")

if __name__ == "__main__":
    build_and_save_index()
