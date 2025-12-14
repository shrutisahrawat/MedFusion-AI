import sys
from pathlib import Path
import streamlit as st

# Ensure Backend is importable
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

# ---------------------- Imports ----------------------
from Backend.RAG.retriever import answer_pubmed_question, PubMedRetriever
from Backend.safety.guards import safety_input_filter, sanitize_output
from Backend.llm.llama_client import generate_text_rag_answer


# ---------------------- UI ----------------------
st.title("📚 PubMed Research Assistant")
st.caption("Ask medical questions and receive PubMed-grounded responses. (Educational use only)")

# Quick check if index exists
retriever = PubMedRetriever()
if not retriever.is_ready():
    st.error("❌ PubMed index not found.\nPlease run the index builder before using this tool.")
    st.stop()


# ---------------------- Main Input ----------------------
user_query = st.text_input(
    "Ask a medical research question:",
    placeholder="e.g., What does PubMed say about pneumonia complications?"
)

ask_btn = st.button("Search & Generate Answer")


# ---------------------- Processing ----------------------
if ask_btn:
    if not user_query.strip():
        st.warning("Please enter a question.")
        st.stop()

    # Safety check BEFORE doing any work
    blocked = safety_input_filter(user_query)
    if blocked:
        st.warning(blocked)
        st.stop()

    with st.spinner("Retrieving PubMed evidence..."):
        results = retriever.retrieve(user_query, top_k=5)

    if not results:
        st.error("❌ No relevant PubMed passages found.")
        st.stop()

    # Extract retrieved chunks only
    context_chunks = [chunk for chunk, dist in results]

    # Display retrieved evidence
    with st.expander("📖 Retrieved PubMed Context Snippets"):
        for i, chunk in enumerate(context_chunks, start=1):
            st.markdown(f"**Context {i}:**\n{chunk}\n---")

    # Generate LLM answer
    with st.spinner("Generating grounded explanation..."):
        llm_answer = generate_text_rag_answer(context_chunks, user_query)
        safe_output = sanitize_output(llm_answer)

    st.subheader("🧠 AI Answer (PubMed-Grounded)")
    st.markdown(safe_output)


# ---------------------- Follow-Up Questions (Late Fusion Extension) ----------------------
st.divider()
st.subheader("💬 Ask a Follow-up Question")

follow_q = st.text_input(
    "Enter follow-up question:",
    placeholder="e.g., What specialists deal with this condition?"
)

if st.button("Ask Follow-up"):
    if not follow_q.strip():
        st.warning("Please type a follow-up question.")
        st.stop()

    blocked2 = safety_input_filter(follow_q)
    if blocked2:
        st.warning(blocked2)
        st.stop()

    with st.spinner("Retrieving updated PubMed results..."):
        new_results = retriever.retrieve(follow_q, top_k=5)

    follow_chunks = [c for c, d in new_results]

    with st.expander("📖 Evidence Snippets Used"):
        for i, chunk in enumerate(follow_chunks, 1):
            st.markdown(f"**Context {i}:**\n{chunk}\n---")

    # Answer using RAG + LLM
    with st.spinner("Generating explanation..."):
        response = generate_text_rag_answer(follow_chunks, follow_q)
        safe_follow = sanitize_output(response)

    st.markdown("### 🧠 Follow-up Answer")
    st.markdown(safe_follow)
