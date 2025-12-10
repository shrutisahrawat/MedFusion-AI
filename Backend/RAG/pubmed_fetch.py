# backend/rag/pubmed_fetch.py

import requests
from xml.etree import ElementTree as ET
import re
from typing import List
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. Fetch PubMed articles by keyword
def fetch_pubmed_ids(term: str, max_results=10) -> List[str]:
    print(f"[fetch_pubmed_ids] Searching PubMed for term: '{term}' with max {max_results} results")
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": term,
        "retmax": max_results,
        "retmode": "xml",
    }
    response = requests.get(base_url, params=params)
    tree = ET.fromstring(response.content)
    ids = [id_elem.text for id_elem in tree.findall(".//Id")]
    print(f"[fetch_pubmed_ids] Found {len(ids)} IDs")
    return ids

def fetch_pubmed_article(pmid: str) -> dict:
    print(f"[fetch_pubmed_article] Fetching article for PMID: {pmid}")
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml",
    }
    response = requests.get(base_url, params=params)
    tree = ET.fromstring(response.content)
    article = {}

    # Extract Title
    title_elem = tree.find(".//ArticleTitle")
    article["title"] = title_elem.text if title_elem is not None else ""

    # Extract Abstract (may have multiple sections)
    abstract_text = []
    for abstr in tree.findall(".//AbstractText"):
        if abstr.text:
            abstract_text.append(abstr.text)
    article["abstract"] = " ".join(abstract_text)
    print(f"[fetch_pubmed_article] Retrieved article title length: {len(article['title'])}, abstract length: {len(article['abstract'])}")
    return article

# 2. Clean & chunk text into ~384 tokens (approximate)
def clean_text(text: str) -> str:
    clean = re.sub(r"<[^>]+>", "", text)
    clean = re.sub(r"\s+", " ", clean)
    return clean.strip()

def chunk_text(text: str, max_tokens: int = 384) -> List[str]:
    sentences = re.split(r"(?<=[.!?]) +", text)
    chunks = []
    chunk = ""
    for sentence in sentences:
        if len(chunk.split()) + len(sentence.split()) <= max_tokens:
            chunk += sentence + " "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + " "
    if chunk:
        chunks.append(chunk.strip())
    return chunks

# 3. Embed and index chunks with FAISS
def embed_text_chunks(chunks: List[str], model_name="sentence-transformers/all-MiniLM-L6-v2"):
    print(f"[embed_text_chunks] Loading model '{model_name}' and encoding {len(chunks)} chunks")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, convert_to_numpy=True)
    print(f"[embed_text_chunks] Completed embeddings with shape {embeddings.shape}")
    return embeddings

def create_faiss_index(embeddings: np.ndarray):
    dimension = embeddings.shape[1]
    print(f"[create_faiss_index] Creating FAISS index with dimension {dimension}")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"[create_faiss_index] Index now contains {index.ntotal} vectors")
    return index

if __name__ == "__main__":
    print("Starting pubmed_fetch.py script...")

    term = "COVID-19 vaccine"
    pmids = fetch_pubmed_ids(term, max_results=5)

    all_chunks = []
    for pmid in pmids:
        article = fetch_pubmed_article(pmid)
        text = article["title"] + ". " + article["abstract"]
        clean = clean_text(text)
        chunks = chunk_text(clean, max_tokens=384)
        print(f"[main] Extracted {len(chunks)} chunks from PMID: {pmid}")
        all_chunks.extend(chunks)

    print(f"[main] Total chunks extracted: {len(all_chunks)}")

    if len(all_chunks) == 0:
        print("[main] No chunks extracted, exiting")
        exit(1)

    embeddings = embed_text_chunks(all_chunks)

    index = create_faiss_index(embeddings)

    query = "Effectiveness of COVID-19 vaccines"
    print(f"[main] Searching for query: '{query}'")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_vec = model.encode([query])
    D, I = index.search(query_vec, k=3)
    print(f"[main] Top 3 matching chunks indices: {I}")
    for idx in I[0]:
        print(f"\nChunk: {all_chunks[idx]}")

    print("Script finished.")
