# backend/llm/prompt.py

"""
Prompt builder utilities for MedFusion AI.

These functions build structured prompts for:
- Text RAG (PubMed)
- Vision (ChestMNIST + FractureMNIST3D)
- PDF report understanding

They are used by backend/llm/llama_client.py.
"""

DEFAULT_DISCLAIMER = (
    "⚠️ This system is for assistance and knowledge purposes only.\n"
    "It is NOT a doctor and may make mistakes.\n"
    "Please consult a qualified doctor or medical professional for real medical advice."
)

BASE_SYSTEM_PROMPT = f"""
You are an AI assistant in a student research project called MedFusion AI.

You are NOT a doctor and this system is NOT a medical device or diagnostic tool.

General rules:
- Use only the provided context: PubMed snippets, vision model outputs, or report text.
- Explain findings in simple, non-technical language for laypersons.
- Do NOT make a definitive diagnosis.
- Do NOT suggest specific medicines, brand names, or exact dosages.
- Do NOT override or contradict doctors, lab reports, or imaging reports.
- When unsure, say you are not certain and recommend consulting a qualified doctor.
- Always encourage users to talk to a real doctor for final decisions.

Always behave cautiously and conservatively.
""".strip()


# ---------- TEXT RAG (PubMed) ----------

def build_text_rag_prompt(context_chunks, user_question: str) -> str:
    """
    context_chunks: list of strings (chunks of PubMed text)
    user_question: the original user query
    """
    context_block = ""
    for i, chunk in enumerate(context_chunks, start=1):
        context_block += f"[CONTEXT {i}]\n{chunk}\n\n"

    user_prompt = f"""
You are given several context snippets from PubMed articles.

{context_block}

USER QUESTION:
{user_question}

TASK:
- Answer the question using ONLY the information from the context snippets above.
- Summarize relevant points in simple, understandable terms.
- Clearly mention that you are not giving a diagnosis or prescription.
- If the context does not fully answer the question, say that more information or a doctor's evaluation is needed.
- Suggest what type of doctor/specialist could be consulted, if appropriate.
- Do not list or recommend specific drug names or dosages.
"""
    return user_prompt.strip()


# ---------- VISION (ChestMNIST / FractureMNIST3D) ----------

def build_vision_prompt(
    chest_summary: str | None,
    fracture_summary: str | None,
    user_description: str | None,
) -> str:
    """
    chest_summary: text summarising model's chest findings (labels + probs)
    fracture_summary: text summarising model's fracture findings
    user_description: optional free-text description from user
    """

    vision_block = "Vision model outputs (from simulated MedMNIST-based models):\n"
    if chest_summary:
        vision_block += f"- Chest X-ray model summary: {chest_summary}\n"
    if fracture_summary:
        vision_block += f"- Fracture scan model summary: {fracture_summary}\n"

    if user_description:
        vision_block += f"\nAdditional description from user:\n{user_description}\n"

    user_prompt = f"""
{vision_block}

TASK:
- Explain these model outputs in general, high-level terms.
- Use careful language: these are only patterns seen by a small model trained on a limited dataset.
- Do NOT treat this as a diagnosis, even if the probabilities are high.
- Do NOT recommend medications, exact dosages, or specific treatment protocols.
- You may suggest general next steps, like "talk to a radiologist", "consult an orthopedician", or "consult a pulmonologist", etc.
- You may suggest gentle, generic lifestyle measures (sleep, hydration, posture, breathing exercises, etc.) but keep them non-specific.
- Explicitly remind the user to consult a doctor and not rely only on this system.
"""
    return user_prompt.strip()


# ---------- PDF REPORT ----------

def build_pdf_prompt(report_text: str, user_question: str | None) -> str:
    """
    report_text: full extracted report text (possibly OCR, noisy)
    user_question: optional user question about the report
    """
    if not user_question or not user_question.strip():
        user_question = (
            "Explain this medical report in simple language, including what body parts are involved and "
            "what these findings might generally indicate."
        )

    # truncate long report for safety
    truncated_report = report_text[:6000]

    user_prompt = f"""
You are given text extracted from a medical report (it may contain noise or OCR errors):

[REPORT TEXT START]
{truncated_report}
[REPORT TEXT END]

USER QUESTION:
{user_question}

TASK:
- Identify and explain the key findings in simple, layperson-friendly language.
- Clarify what body parts, organs, or systems are being discussed.
- Explain general possible implications without calling it a confirmed diagnosis.
- Do NOT suggest exact medicines, dosages, or treatment plans.
- Encourage the user to ask the referring doctor or a relevant specialist to interpret the report fully.
"""
    return user_prompt.strip()
