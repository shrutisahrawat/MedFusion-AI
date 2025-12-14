import streamlit as st

st.set_page_config(
    page_title="MedFusion-AI",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 MedFusion-AI")
st.subheader("Multimodal Medical AI Assistant (Educational Use Only)")

st.info(
    "⚠️ This system is for educational and research purposes only.\n"
    "It does NOT diagnose diseases or prescribe treatments.\n"
    "Always consult a qualified medical professional."
)

st.markdown("""
### Available Modules
- 🖼️ **Image Assistant** – X-ray / CT based AI models
- 📄 **Report Assistant** – Medical PDF explanation
- 📚 **Text RAG Assistant** – PubMed-based knowledge assistant

👉 Use the **sidebar** to navigate.
""")
