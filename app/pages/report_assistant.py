import sys
from pathlib import Path
import tempfile
import streamlit as st

# Root directory for imports
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

# ---------------------- Backend imports ----------------------
from Backend.pdf.extract_text import extract_text_from_pdf
from Backend.safety.guards import safety_input_filter, sanitize_output
from Backend.llm.llama_client import generate_pdf_answer
from Backend.RAG.retriever import answer_pubmed_question  # for follow-up questions


# ---------------------- Page Title ----------------------
st.title("📄 Medical Report Assistant")
st.caption("Upload a medical PDF and get a simple, safe explanation (NOT a diagnosis).")


# ---------------------- PDF Upload ----------------------
uploaded = st.file_uploader("Upload a medical report (PDF)", type=["pdf"])

if uploaded:
    # Save PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.read())
        pdf_path = Path(tmp.name)

    st.info("Extracting text from the PDF...")
    report_text = extract_text_from_pdf(pdf_path)

    if not report_text.strip():
        st.error("❌ No readable text found. It may be a scanned PDF without OCR.")
        st.stop()

    # Safety filter BEFORE using the LLM
    blocked = safety_input_filter(report_text)
    if blocked:
        st.warning(blocked)
        st.stop()

    # Preview extracted text
    with st.expander("📄 Preview Extracted Report Text"):
        st.text(report_text[:4000])

    # ---------------------- Explanation Button ----------------------
    if st.button("Generate Explanation"):
        with st.spinner("Analyzing report…"):
            explanation = generate_pdf_answer(report_text, user_question=None)

        safe_explanation = sanitize_output(explanation)

        st.subheader("🧾 AI Explanation (Safe Mode)")
        st.markdown(safe_explanation)

        st.warning(
            "⚠️ This explanation is for educational purposes only.\n"
            "This system is NOT a doctor. Always consult a licensed medical professional."
        )

    # ---------------------- Follow-Up Q&A (Late Fusion with RAG) ----------------------
    st.subheader("💬 Ask Follow-up Questions About This Report")

    user_q = st.text_input("Your question about the report:")

    if st.button("Ask Question"):
        if not user_q.strip():
            st.error("Please enter a question.")
            st.stop()

        # Safety on follow-up question
        blocked_q = safety_input_filter(user_q)
        if blocked_q:
            st.warning(blocked_q)
            st.stop()

        with st.spinner("Retrieving PubMed evidence + generating response…"):

            # PubMed RAG retrieval
            rag_evidence = answer_pubmed_question(user_q)

            # Late Fusion: combine report text + RAG + user question
            combined_context = (
                f"REPORT EXTRACT:\n{report_text[:2000]}\n\n"
                f"PUBMED EVIDENCE:\n{rag_evidence}\n\n"
                f"USER QUESTION:\n{user_q}"
            )

            final_answer = generate_pdf_answer(
                report_text=combined_context,
                user_question=user_q
            )

            safe_final = sanitize_output(final_answer)

        st.subheader("📚 Evidence-Based Answer")
        st.markdown(safe_final)
