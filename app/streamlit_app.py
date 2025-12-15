import streamlit as st

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="MedFusion-AI",
    page_icon="🧠",
    layout="wide"
)

# ---------------------- HEADER ----------------------
st.title("🧠 MedFusion-AI")
st.subheader("A Multimodal Medical AI Assistant (Educational & Research Use Only)")

st.info(
    "⚠️ **Important Safety Notice**\n"
    "- This system is built for *learning and research*. It is **NOT a medical device**.\n"
    "- Vision and text models may produce inaccurate results; do **not** rely on them for diagnosis.\n"
    "- Always consult a qualified medical professional for actual medical concerns."
)

# ---------------------- INTRODUCTION ----------------------
st.markdown("""
MedFusion-AI integrates **three major AI pipelines** into a unified assistant that combines
**medical literature retrieval**, **vision-based pattern recognition**, and **local LLM reasoning**.

---

## 🔍 1. **PubMed Research Assistant (Text RAG)**  
Uses:
- PubMed article retrieval  
- Text cleaning, chunking, and embedding  
- FAISS vector search  
- Llama-3.1-8B (via Ollama) for grounded explanations  

This ensures answers are **evidence-backed**, conversational, and safe.

---

## 🖼️ 2. **Image Assistant (MedMNIST Vision Models)**  
Performs non-diagnostic, educational image interpretation using custom-trained **ResNet-18** models on:
- ChestMNIST  
- PneumoniaMNIST  
- BreastMNIST  
- OrganAMNIST  

The vision output is converted into natural-language summaries  
and then explained by the LLM.  
Follow-up questions trigger **late fusion** with PubMed for deeper insights.

---

## 📄 3. **PDF Report Assistant**  
Extracts text from medical PDFs (OCR-friendly) and provides:
- Simple, safe explanations  
- Highlights of key findings  
- Optional follow-up Q&A with PubMed RAG integration  

Designed for academic understanding of radiology & pathology reports.

---

## 🚀 **How to Use This App**
Use the **left sidebar** to navigate between:
- 🖼️ **Image Assistant**  
- 📚 **Text RAG Assistant**  
- 📄 **PDF Report Assistant**

Each module follows the same principles:
- Retrieval-augmented context  
- Local LLM reasoning  
- Strict safety & non-diagnostic constraints  
- Friendly, simplified explanations  

---

""")

# ---------------------- FOOTER ----------------------
st.markdown("---")
st.caption(
    "© MedFusion-AI — Student Research Project | Combines PubMed RAG, MedMNIST Vision Models, "
    "and Local LLMs for Safe Educational Assistance."
)
