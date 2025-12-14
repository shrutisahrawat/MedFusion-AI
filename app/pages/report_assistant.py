import shutil
import streamlit as st

import streamlit as st
import tempfile
from pathlib import Path

from Backend.pdf.extract_text import extract_text_from_pdf
from Backend.safety.guards import (
    safety_input_filter,
    redact_phi,
    trim_text,
    sanitize_output,
)
from Backend.llm.prompt import medical_report_prompt
from Backend.llm.llama_client import call_llama_ollama  # adjust if needed

st.title("📄 Medical Report Assistant")
st.caption("Upload a medical PDF for explanation (no diagnosis).")

uploaded = st.file_uploader("Upload PDF Report", type=["pdf"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.read())
        pdf_path = Path(tmp.name)

    st.info("Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)

    if not text.strip():
        st.error("No readable text found (scanned PDF).")
        st.stop()

    blocked = safety_input_filter(text)
    if blocked:
        st.warning(blocked)
        st.stop()

    clean_text = redact_phi(text)
    clean_text, trimmed = trim_text(clean_text)

    with st.expander("Preview extracted text"):
        st.text(clean_text[:3000])

    if st.button("Generate Explanation"):
        with st.spinner("Generating explanation..."):
            response = call_llama_ollama(
                medical_report_prompt(clean_text)
            )

        safe_response = sanitize_output(response)

        st.subheader("🧾 Explanation")
        st.markdown(safe_response)

        st.warning(
            "⚠️ Educational explanation only. "
            "Consult a medical professional for advice."
        )
