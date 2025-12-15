import sys
from pathlib import Path
import tempfile
import streamlit as st

# Root directory for imports
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

# ---------------------- Backend imports ----------------------
from Backend.pdf.extract_text import extract_text_from_pdf
from Backend.safety.guards import safety_input_filter, redact_phi, trim_text, sanitize_output
from Backend.llm.llama_client import generate_pdf_answer
from Backend.RAG.retriever import answer_pubmed_question


# ---------------------- Page Title ----------------------
st.title("📄 Medical Report Assistant")
st.caption("Upload a medical PDF and get a simple, safe explanation. (NOT a diagnosis)")

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

    # Safety filter BEFORE sending to LLM
    blocked = safety_input_filter(report_text)
    if blocked:
        st.warning(blocked)
        st.stop()

    # Redact PHI
    clean_text = redact_phi(report_text)
    clean_text, was_trimmed = trim_text(clean_text)

    # Preview extracted text
    with st.expander("📄 Preview Extracted Report Text"):
        st.text(clean_text[:4000])

    # ---------------------- Explanation Button ----------------------
    if st.button("Generate Explanation"):
        with st.spinner("Analyzing report…"):
            explanation = generate_pdf_answer(clean_text, user_question=None)

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

        with st.spinner("Retrieving research evidence…"):
            rag_output = answer_pubmed_question(user_q)

        if isinstance(rag_output, dict):
            pubmed_answer = rag_output.get("answer", "")
            contexts = rag_output.get("contexts", [])
        else:
            pubmed_answer = str(rag_output)
            contexts = []

        # Build fusion-style context
        combined_context = (
            f"EXTRACTED REPORT TEXT:\n{clean_text[:2000]}\n\n"
            f"PUBMED RESEARCH SUMMARY:\n{pubmed_answer}\n\n"
            f"RELEVANT CONTEXTS:\n{contexts}\n\n"
            f"USER QUESTION:\n{user_q}"
        )

        # LLM final answer
        with st.spinner("Generating final explanation…"):
            final = generate_pdf_answer(report_text=combined_context, user_question=user_q)
            safe_final = sanitize_output(final)

        st.subheader("📚 Evidence-Based Answer")
        st.markdown(safe_final)
