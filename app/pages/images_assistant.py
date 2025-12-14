import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

import streamlit as st
from PIL import Image

# Vision predictions + summaries
from Backend.vision.inference import (
    predict_chest_from_pil, summarize_chest_results,
    predict_pneumonia_from_pil, summarize_pneumonia_results,
    predict_breast_from_pil, summarize_breast_results,
    predict_organ_from_pil, summarize_organ_results,
)

# LLM + RAG
from Backend.llm.llama_client import generate_vision_answer
from Backend.RAG.retriever import answer_pubmed_question


st.title("🩻 Image Assistant — Multimodal Medical AI")
st.caption("Educational AI system combining image classification + LLM reasoning + PubMed RAG.")

# Session memory
for key in ["vision_summary", "vision_raw", "initial_explanation", "image_uploaded"]:
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

    user_note = st.text_area("Optional: add symptoms or comments")

    if st.button("Analyze Image"):
        with st.spinner("Analyzing image…"):

            # Vision step
            if model_choice.startswith("Chest"):
                pred = predict_chest_from_pil(image)
                summary = summarize_chest_results(pred)

            elif model_choice.startswith("Pneumonia"):
                pred = predict_pneumonia_from_pil(image)
                summary = summarize_pneumonia_results(pred)

            elif model_choice.startswith("Breast"):
                pred = predict_breast_from_pil(image)
                summary = summarize_breast_results(pred)

            elif model_choice.startswith("Organ"):
                pred = predict_organ_from_pil(image)
                summary = summarize_organ_results(pred)

            st.session_state.vision_raw = pred
            st.session_state.vision_summary = summary

            # LLM Explanation
            explanation = generate_vision_answer(
                chest_summary=summary if model_choice.startswith("Chest") else None,
                fracture_summary=None,
                user_description=user_note,
            )

            st.session_state.initial_explanation = explanation

    # Show results
    if st.session_state.vision_raw:
        st.subheader("🔍 Model Output (Raw)")
        st.json(st.session_state.vision_raw)

    if st.session_state.initial_explanation:
        st.subheader("🧠 AI Explanation (LLM)")
        st.write(st.session_state.initial_explanation)


# -------------------- FOLLOW-UP QUESTIONS (LATE FUSION) --------------------
if st.session_state.image_uploaded and st.session_state.vision_summary:
    st.subheader("💬 Ask Follow-Up Questions")

    followup = st.text_input("Your question:")

    if st.button("Ask"):
        with st.spinner("Retrieving PubMed evidence…"):

            rag_answer = answer_pubmed_question(followup)

            fused_text = (
                f"Vision model findings: {st.session_state.vision_summary}\n\n"
                f"PubMed evidence:\n{rag_answer}\n\n"
                f"User question: {followup}"
            )

            final_answer = generate_vision_answer(
                chest_summary=st.session_state.vision_summary,
                fracture_summary=None,
                user_description=fused_text
            )

        st.subheader("📚 Evidence-Based AI Response")
        st.write(final_answer)
