import sys
from pathlib import Path
import streamlit as st

# --------------------------------------------------
# Make Backend importable
# --------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

# --------------------------------------------------
# Imports
# --------------------------------------------------
from Backend.RAG.retriever import PubMedRetriever
from Backend.safety.guards import safety_input_filter, sanitize_output
from Backend.llm.llama_client import generate_text_rag_answer


# ==================================================
# HELPER: merge retriever outputs safely
# ==================================================
def merge_retrieved_evidence(result_dict, top_k=6):
    """
    Merge definition / mechanism / research evidence
    into a single clean list for the LLM.
    """
    merged = []

    for key in ("definition_support", "mechanism_support", "research_support"):
        merged.extend(result_dict.get(key, []))

    # Deduplicate by PMID + text
    seen = set()
    cleaned = []

    for rec in merged:
        pmid = rec.get("pmid", "N/A")
        text = rec.get("text", "").strip()

        sig = (pmid, text)
        if sig not in seen and text:
            cleaned.append({"pmid": pmid, "text": text})
            seen.add(sig)

    return cleaned[:top_k]


# ==================================================
# UI
# ==================================================
st.set_page_config(page_title="PubMed Research Assistant", layout="centered")

st.title("📚 PubMed Research Assistant")
st.caption(
    "Evidence-based medical explanations using PubMed research "
    "(Educational use only — NOT a diagnosis)."
)

retriever = PubMedRetriever()


# ==================================================
# MAIN QUESTION
# ==================================================
user_query = st.text_input(
    "Ask a medical research question:",
    placeholder="e.g., What is asthma?"
)

if st.button("Search & Explain"):

    if not user_query.strip():
        st.warning("Please enter a question.")
        st.stop()

    # Safety check
    blocked = safety_input_filter(user_query)
    if blocked:
        st.warning(blocked)
        st.stop()

    # ---------------- Retrieval ----------------
    with st.spinner("🔎 Retrieving relevant PubMed evidence..."):
        retrieval = retriever.retrieve(user_query)

    context_records = merge_retrieved_evidence(retrieval)

    if not context_records:
        st.error("❌ No relevant PubMed evidence found.")
        st.stop()

    # ---------------- Evidence Display ----------------
    with st.expander("📖 PubMed Evidence Used"):
        for i, rec in enumerate(context_records, start=1):
            st.markdown(
                f"**Snippet {i} — PMID {rec['pmid']}**\n\n"
                f"{rec['text']}\n\n---"
            )

    # ---------------- LLM Explanation ----------------
    with st.spinner("🧠 Generating evidence-based explanation..."):
        answer = generate_text_rag_answer(
            context_records=context_records,
            user_question=user_query
        )

    st.subheader("🧠 AI Explanation (PubMed-Grounded)")
    st.markdown(sanitize_output(answer))


# ==================================================
# FOLLOW-UP QUESTIONS
# ==================================================
st.divider()
st.subheader("💬 Ask a Follow-Up Question")

follow_q = st.text_input(
    "Enter a follow-up question:",
    placeholder="e.g., What complications of asthma are reported in PubMed?"
)

if st.button("Ask Follow-Up"):

    if not follow_q.strip():
        st.warning("Please enter a follow-up question.")
        st.stop()

    blocked = safety_input_filter(follow_q)
    if blocked:
        st.warning(blocked)
        st.stop()

    # ---------------- Retrieval ----------------
    with st.spinner("🔎 Retrieving updated PubMed evidence..."):
        retrieval = retriever.retrieve(follow_q)

    follow_records = merge_retrieved_evidence(retrieval)

    if not follow_records:
        st.error("❌ No relevant PubMed evidence found.")
        st.stop()

    # ---------------- Evidence Display ----------------
    with st.expander("📖 PubMed Evidence Used"):
        for i, rec in enumerate(follow_records, start=1):
            st.markdown(
                f"**Snippet {i} — PMID {rec['pmid']}**\n\n"
                f"{rec['text']}\n\n---"
            )

    # ---------------- LLM Answer ----------------
    with st.spinner("🧠 Generating explanation..."):
        follow_answer = generate_text_rag_answer(
            context_records=follow_records,
            user_question=follow_q
        )

    st.subheader("🧠 Follow-Up Answer")
    st.markdown(sanitize_output(follow_answer))
