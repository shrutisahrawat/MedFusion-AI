# Backend/llm/prompt.py
from typing import Optional

"""
Prompt builder utilities for MedFusion AI.

These functions build structured prompts for:
- Text RAG (PubMed)
- Vision (ChestMNIST / BreastMNIST / PneumoniaMNIST / OrganAMNIST)
- PDF report understanding

This file is fully Python 3.9 compatible and follows strict medical safety rules.
"""

# ---------------- GLOBAL SAFETY DISCLAIMER ----------------

DEFAULT_DISCLAIMER = (
    "⚠️ This system is for educational and assistance purposes only.\n"
    "It is NOT a doctor and may make mistakes.\n"
    "It does NOT provide medical diagnosis, prescriptions, or treatment plans.\n"
    "Always consult a qualified doctor or healthcare professional for real medical advice."
)

BASE_SYSTEM_PROMPT = f"""
You are an AI assistant in a student research project called MedFusion AI.

You are NOT a doctor and this system is NOT a certified medical device.

STRICT SAFETY RULES (MUST ALWAYS FOLLOW):
- Provide information ONLY for education and general awareness.
- Do NOT provide medical diagnosis or confirmation of disease.
- Do NOT prescribe medicines, drug names, injections, or dosages.
- Do NOT provide emergency medical instructions.
- Do NOT assist with unsafe medical practices.
- Always remind users to consult a qualified doctor for final decisions.
- If uncertain, clearly say you are not sure.

COMMUNICATION STYLE:
- Simple, calm, respectful, and non-alarming.
- Layperson-friendly language.
- Conservative and cautious tone.

{DEFAULT_DISCLAIMER}
""".strip()


# ---------------- TEXT RAG (PUBMED) ----------------

def build_text_rag_prompt(context_chunks, user_question: str) -> str:
    """
    context_chunks: list of strings (chunks of PubMed text)
    user_question: the original user query
    """
    context_block = ""
    for i, chunk in enumerate(context_chunks, start=1):
        context_block += f"[CONTEXT {i}]\n{chunk}\n\n"

    user_prompt = f"""
You are given multiple scientific context snippets from PubMed articles.

{context_block}

USER QUESTION:
{user_question}

TASK:
- Answer ONLY using the provided context snippets.
- Summarize relevant facts in simple and easy language.
- Do NOT make a medical diagnosis.
- Do NOT suggest specific medicines, brand names, or drug dosages.
- Clearly remind that this is not medical advice.
- If the context is insufficient, say that medical evaluation is required.
- If appropriate, mention what type of doctor (e.g., pulmonologist, radiologist, oncologist) may be consulted.
"""
    return user_prompt.strip()


# ---------------- VISION (CHEST / BREAST / PNEUMONIA / ORGAN) ----------------

def build_vision_prompt(
    chest_summary: Optional[str],
    breast_summary: Optional[str],
    pneumonia_summary: Optional[str],
    organ_summary: Optional[str],
    user_description: Optional[str],
) -> str:
    """
    chest_summary: summary of ChestMNIST predictions (e.g., chest X-ray findings)
    breast_summary: summary of BreastMNIST predictions (e.g., breast lesion patterns)
    pneumonia_summary: summary of PneumoniaMNIST predictions (e.g., pneumonia vs normal)
    organ_summary: summary of OrganAMNIST predictions (e.g., organ region classification)
    user_description: optional free-text input from user (symptoms, history, etc.)
    """

    vision_block = "Vision model outputs (from educational MedMNIST-based models):\n"

    if chest_summary:
        vision_block += f"- Chest model summary (ChestMNIST): {chest_summary}\n"

    if breast_summary:
        vision_block += f"- Breast model summary (BreastMNIST): {breast_summary}\n"

    if pneumonia_summary:
        vision_block += f"- Pneumonia model summary (PneumoniaMNIST): {pneumonia_summary}\n"

    if organ_summary:
        vision_block += f"- Organ model summary (OrganAMNIST): {organ_summary}\n"

    if user_description:
        vision_block += f"\nAdditional user description:\n{user_description}\n"

    user_prompt = f"""
{vision_block}

TASK:
- Explain what these model outputs MAY indicate in very general terms.
- Emphasize that these are only patterns from small educational AI models trained on MedMNIST subsets.
- They are NOT a diagnosis and may be incorrect.
- Do NOT recommend medicines, surgeries, injections, or detailed treatment plans.
- You may suggest general next steps such as:
  - "Consult a radiologist" (for imaging),
  - "Consult a pulmonologist" (for lung findings),
  - "Consult an oncologist" (for suspected tumors),
  - "Consult a gastroenterologist / general physician" (for abdominal/organ findings).
- You may suggest very general lifestyle advice like rest, hydration, basic posture care, but keep it non-specific.
- Clearly remind users to consult a doctor and not rely only on this system.
"""
    return user_prompt.strip()


# ---------------- PDF MEDICAL REPORT ----------------

def build_pdf_prompt(report_text: str, user_question: Optional[str]) -> str:
    """
    report_text: extracted raw medical report text (OCR may be noisy)
    user_question: optional user question about the report
    """

    if not user_question or not user_question.strip():
        user_question = (
            "Explain this medical report in simple language and describe what body parts are involved."
        )

    # Truncate long reports for safety and token limits
    truncated_report = report_text[:6000]

    user_prompt = f"""
You are given text extracted from a medical report:

[REPORT TEXT START]
{truncated_report}
[REPORT TEXT END]

USER QUESTION:
{user_question}

TASK:
- Explain the main findings in simple, non-technical words.
- Mention which organs or body parts are discussed.
- Explain possible general meaning without calling it a confirmed diagnosis.
- Do NOT suggest medications, dosages, therapies, or treatments.
- Encourage consulting the referring doctor or relevant specialist.
"""
    return user_prompt.strip()
