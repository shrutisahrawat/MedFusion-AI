# Backend/llm/prompt.py
from typing import Optional, List


# =====================================================
# GLOBAL DISCLAIMER
# =====================================================
DEFAULT_DISCLAIMER = (
    "⚠️ This system is for educational assistance only.\n"
    "It is NOT a doctor and may make mistakes.\n"
    "It does NOT diagnose conditions or prescribe treatments.\n"
    "Always consult a qualified medical professional for real medical advice."
)


# =====================================================
# BASE SYSTEM PROMPT
# =====================================================
BASE_SYSTEM_PROMPT = f"""
You are MedFusion-AI — an educational assistant designed to explain medical concepts clearly.

STRICT SAFETY RULES:
- DO NOT diagnose diseases.
- DO NOT prescribe medicines or treatments.
- DO NOT provide emergency instructions.
- State uncertainty when evidence is limited.
- Encourage consulting a medical professional.

COMMUNICATION STYLE:
- Answer like a knowledgeable teacher, not a researcher.
- Start with a clear, simple explanation.
- Use PubMed evidence ONLY to support explanations.
- Avoid listing studies, papers, or case reports.
- Avoid copying or summarizing abstracts.

{DEFAULT_DISCLAIMER}
""".strip()


# =====================================================
# USER-FOCUSED QUESTION GUIDE  🔥 IMPORTANT
# =====================================================
def build_user_focused_question(user_question: str) -> str:
    """
    Guides the LLM to explain concepts clearly and completely
    using PubMed as background support.
    """

    return f"""
Answer the following question for a general, non-medical user.

RESPONSE STRUCTURE (FOLLOW THIS):
1. Start with a clear, direct definition.
2. Explain what happens inside the body (simple biology).
3. Mention common causes (high-level).
4. Mention common symptoms (non-alarming).
5. Briefly mention who may be at higher risk (if relevant).
6. Use PubMed research ONLY to support explanations (typically 2–3 relevant PMIDs).
7. Prefer review articles or consensus-level evidence when available.


IMPORTANT RULES:
- Do NOT list studies or case reports.
- Do NOT summarize abstracts.
- Do NOT mention rare complications unless asked.
- Keep the explanation calm, simple, and educational.

USER QUESTION:
{user_question}
""".strip()



# =====================================================
# 1. PUBMED RAG PROMPT (FIXED)
# =====================================================
def build_pubmed_rag_prompt(
    context_records: List[dict],
    guided_question: str,
    history: Optional[str],
    bookshelf_text: Optional[str] = None,
) -> str:
    """
    Builds a clean, user-facing RAG prompt.
    Bookshelf = foundation
    PubMed = evidence
    """

    history_block = f"\nPrevious conversation:\n{history}\n" if history else ""

    # 📘 Bookshelf (FOUNDATION)
    bookshelf_block = ""
    if bookshelf_text:
        bookshelf_block = f"""
FOUNDATIONAL MEDICAL EXPLANATION (from standard medical references):
{bookshelf_text}
"""

    # 📄 PubMed (SUPPORTING EVIDENCE)
    evidence_lines = []
    seen_pmids = set()

    for rec in context_records:
        pmid = rec.get("pmid")
        text = rec.get("text", "").strip()

        if pmid and pmid not in seen_pmids:
            evidence_lines.append(f"- {text} (PMID {pmid})")
            seen_pmids.add(pmid)

        if len(seen_pmids) >= 3:   # 🔥 force 2–3 PMIDs max
            break

    evidence_block = "\n".join(evidence_lines)

    return f"""
{history_block}

You are answering a medical question for a general user.

{bookshelf_block}

SUPPORTING RESEARCH EVIDENCE (PubMed):
{evidence_block}

{guided_question}

IMPORTANT:
- Explain naturally, like a medical educator.
- Do NOT mention studies, trials, or paper titles.
- Do NOT copy abstracts.
- Use PubMed only to support the explanation.
- End with the disclaimer.

{DEFAULT_DISCLAIMER}
""".strip()



# =====================================================
# 2. VISION EXPLANATION PROMPT
# =====================================================
def build_vision_prompt(
    chest_summary: Optional[str],
    breast_summary: Optional[str],
    pneumonia_summary: Optional[str],
    organ_summary: Optional[str],
    user_description: Optional[str] = None
) -> str:

    blocks = []

    if chest_summary:
        blocks.append(f"Chest X-ray Model: {chest_summary}")
    if breast_summary:
        blocks.append(f"Breast Model: {breast_summary}")
    if pneumonia_summary:
        blocks.append(f"Pneumonia Model: {pneumonia_summary}")
    if organ_summary:
        blocks.append(f"Organ CT Model: {organ_summary}")

    vision_block = "\n".join(f"- {b}" for b in blocks)

    if user_description:
        vision_block += f"\n\nUser notes:\n{user_description}"

    return f"""
The system produced the following educational model outputs:

{vision_block}

TASK:
- Follow the response structure exactly.
- Explain the concept clearly and completely in plain language.
- Use PubMed ONLY as background support and it should be relevant to user question.
- DO NOT describe studies, research designs, cohorts, or findings.
- DO NOT say phrases like:
  “a study found”, “research shows”, “according to a study”.
- If citing PubMed, do so briefly and passively, e.g.:
  “This understanding is supported by medical literature (PMID XXXXX).”
- Mention at most THREE PMID for definition-style questions.
- Focus on understanding, not research analysis.
- End with the disclaimer ONCE.



{DEFAULT_DISCLAIMER}
""".strip()


# =====================================================
# 3. LATE FUSION PROMPT (VISION + PUBMED)
# =====================================================
def build_fusion_prompt(
    vision_summary: str,
    context_records: List[dict],
    user_question: str,
    history: Optional[str] = None
) -> str:

    evidence = ""
    for rec in context_records:
        pmid = rec.get("pmid", "N/A")
        text = rec.get("text", "").strip()
        evidence += f"- PMID {pmid}: {text}\n"

    history_block = f"\nPREVIOUS CONVERSATION:\n{history}\n" if history else ""

    return f"""
{history_block}

VISION SUMMARY:
{vision_summary}

PUBMED BACKGROUND (supporting only):
{evidence}

USER QUESTION:
{user_question}

TASK:
- Combine vision insights with research knowledge.
- Explain in simple language.
- Do NOT list studies or abstracts.
- Mention PMIDs sparingly and naturally.
- Avoid diagnosis or treatment advice.

{DEFAULT_DISCLAIMER}
""".strip()


# =====================================================
# 4. PDF REPORT PROMPT
# =====================================================
def build_pdf_prompt(report_text: str, user_question: Optional[str]) -> str:

    if not user_question or not user_question.strip():
        user_question = "Explain this medical report in simple language."

    truncated = report_text[:6000]

    return f"""
You are given a medical report:

[REPORT START]
{truncated}
[REPORT END]

USER QUESTION:
{user_question}

TASK:
- Explain the report in simple terms.
- Describe which organs or systems are involved.
- Do NOT diagnose or suggest treatment.

{DEFAULT_DISCLAIMER}
""".strip()
