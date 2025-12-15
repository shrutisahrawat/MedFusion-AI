import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

import streamlit as st
from PIL import Image

# Vision model predictions + summaries
from Backend.vision.inference import (
    predict_chest_from_pil, summarize_chest_results,
    predict_pneumonia_from_pil, summarize_pneumonia_results,
    predict_breast_from_pil, summarize_breast_results,
    predict_organ_from_pil, summarize_organ_results,
)

# LLM + RAG
from Backend.llm.llama_client import (
    generate_vision_answer,
    generate_fusion_answer,
)
from Backend.RAG.retriever import answer_pubmed_question
from Backend.safety.guards import sanitize_output, safety_input_filter


st.title("🩻 Image Assistant — Multimodal Medical AI")
st.caption("Educational system combining MedMNIST vision models + LLM reasoning + PubMed RAG.")

# -------------------- SESSION MEMORY --------------------
for key in ["vision_summary", "vision_raw", "llm_explanation", "image_uploaded"]:
    if key not in st.session_state:
        st.session_state[key] = None


# -------------------- UI --------------------
model_choice = st.selectbox(
    "Choose a Vision Model",
    [
        "ChestMNIST (Multi-label X-ray)",
        "Pneumonia Detection",
        "Breast Cancer Classification",
        "Organ Classification (CT)"
    ]
)

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])


# -------------------- RUN ANALYSIS --------------------
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.session_state.image_uploaded = True

    user_note = st.text_area("Optional: Add symptoms, history, or comments")

    if st.button("Analyze Image"):
        with st.spinner("Running vision model…"):

            # ========== VISION MODEL STEP ==========
            if model_choice.startswith("Chest"):
                pred = predict_chest_from_pil(image)
                summary = summarize_chest_results(pred)
                chest_s, pneumonia_s, breast_s, organ_s = summary, None, None, None

            elif model_choice.startswith("Pneumonia"):
                pred = predict_pneumonia_from_pil(image)
                summary = summarize_pneumonia_results(pred)
                chest_s, pneumonia_s, breast_s, organ_s = None, summary, None, None

            elif model_choice.startswith("Breast"):
                pred = predict_breast_from_pil(image)
                summary = summarize_breast_results(pred)
                chest_s, pneumonia_s, breast_s, organ_s = None, None, summary, None

            elif model_choice.startswith("Organ"):
                pred = predict_organ_from_pil(image)
                summary = summarize_organ_results(pred)
                chest_s, pneumonia_s, breast_s, organ_s = None, None, None, summary

            # Store session memory
            st.session_state.vision_raw = pred
            st.session_state.vision_summary = summary

            # ========== LLM EXPLANATION ==========
            llm_response = generate_vision_answer(
                chest_summary=chest_s,
                breast_summary=breast_s,
                pneumonia_summary=pneumonia_s,
                organ_summary=organ_s,
                user_description=user_note,
            )

            st.session_state.llm_explanation = sanitize_output(llm_response)

    # -------------------- SHOW RESULTS --------------------
    if st.session_state.vision_raw:
        st.subheader("🔍 Raw Model Output")
        st.json(st.session_state.vision_raw)

    if st.session_state.llm_explanation:
        st.subheader("🧠 AI Explanation")
        st.write(st.session_state.llm_explanation)


# -------------------- FOLLOW-UP QUESTIONS (LATE FUSION) --------------------
if st.session_state.image_uploaded and st.session_state.vision_summary:
    st.subheader("💬 Ask a Follow-Up Question (Vision + PubMed RAG)")

    follow = st.text_input("Your follow-up question:")

    if st.button("Ask Follow-Up"):
        if not follow.strip():
            st.warning("Please enter a question.")
            st.stop()

        blocked = safety_input_filter(follow)
        if blocked:
            st.warning(blocked)
            st.stop()

        with st.spinner("Retrieving PubMed evidence…"):
            rag = answer_pubmed_question(follow)

        # Extract contexts & pmids from RAG result
        if isinstance(rag, dict):
            pmids = rag.get("pmids", [])
            context_records = rag.get("contexts", [])   # <-- already {text, pmid}
        else:
            pmids = []
            context_records = []

        # ----- FINAL FUSION ANSWER -----
        with st.spinner("Combining vision + PubMed evidence…"):

            fusion_answer = generate_fusion_answer(
                vision_summary=st.session_state.vision_summary,
                context_records=context_records,
                user_question=follow,
            )

            fusion_safe = sanitize_output(fusion_answer)

        st.subheader("📚 Evidence-Based Explanation")
        st.write(fusion_safe)
