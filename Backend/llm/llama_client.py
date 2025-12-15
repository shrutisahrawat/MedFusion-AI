# Backend/llm/llama_client.py

import os
from typing import List, Optional
import requests

from Backend.llm.prompt import (
    BASE_SYSTEM_PROMPT,
    build_pubmed_rag_prompt,
    build_user_focused_question,
    build_vision_prompt,
    build_fusion_prompt,
    build_pdf_prompt,
)
from Backend.safety.guards import safety_input_filter, sanitize_output


# =============================================================
# Ollama configuration
# =============================================================
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "800"))
TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.12"))


# =============================================================
# Conversation memory (bounded, clean)
# =============================================================
MAX_TURNS = 4
conversation_history: List[str] = []


def _append_history(user_q: str, assistant_a: str):
    conversation_history.append(f"User: {user_q}\nAssistant: {assistant_a}")
    if len(conversation_history) > MAX_TURNS:
        conversation_history.pop(0)


def _history_text() -> str:
    return "\n\n".join(conversation_history)


# =============================================================
# Low-level generator
# =============================================================
def _generate(system_prompt: str, user_prompt: str) -> str:
    full_prompt = (
        f"<|system|>\n{system_prompt}"
        f"\n<|user|>\n{user_prompt}"
        f"\n<|assistant|>\n"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
        },
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
    resp.raise_for_status()

    data = resp.json()
    return (data.get("response") or data.get("text") or "").strip()


# =============================================================
# 1️⃣ PubMed RAG Answer — CLEAN & CORRECT
# =============================================================
def generate_text_rag_answer(
    context_records: List[dict],   # [{"text": ..., "pmid": ...}]
    user_question: str,
    bookshelf_text: Optional[str] = None,
) -> str:
    """
    Clean medical explanation:
    - Explanation first
    - PubMed supports claims
    - 2–3 PMIDs max
    - NO prompt leakage
    """

    # Safety
    blocked = safety_input_filter(user_question)
    if blocked:
        return blocked

    # ----------------------------------
    # Prepare PubMed evidence cleanly
    # ----------------------------------
    evidence_records = []
    seen_pmids = set()

    for rec in context_records:
        pmid = rec.get("pmid")
        text = rec.get("text", "").strip()

        if pmid and pmid not in seen_pmids:
            evidence_records.append({
                "pmid": pmid,
                "text": text
            })
            seen_pmids.add(pmid)

        if len(seen_pmids) >= 3:
            break

    # ----------------------------------
    # Build guided question
    # ----------------------------------
    guided_question = build_user_focused_question(user_question)

    # ----------------------------------
    # Build final prompt (NO SEPARATORS)
    # ----------------------------------
    prompt = build_pubmed_rag_prompt(
        context_records=evidence_records,
        guided_question=guided_question,
        history=_history_text(),
        bookshelf_text=bookshelf_text,
    )

    raw = _generate(BASE_SYSTEM_PROMPT, prompt)
    safe = sanitize_output(raw)

    _append_history(user_question, safe)
    return safe


# =============================================================
# 2️⃣ Vision-only Explanation
# =============================================================
def generate_vision_answer(
    chest_summary: Optional[str],
    breast_summary: Optional[str],
    pneumonia_summary: Optional[str],
    organ_summary: Optional[str],
    user_description: Optional[str],
) -> str:

    combined = " ".join(
        x for x in [
            chest_summary,
            breast_summary,
            pneumonia_summary,
            organ_summary,
            user_description,
        ] if x
    )

    blocked = safety_input_filter(combined)
    if blocked:
        return blocked

    prompt = build_vision_prompt(
        chest_summary,
        breast_summary,
        pneumonia_summary,
        organ_summary,
        user_description,
    )

    raw = _generate(BASE_SYSTEM_PROMPT, prompt)
    return sanitize_output(raw)


# =============================================================
# 3️⃣ Late Fusion (Vision + PubMed)
# =============================================================
def generate_fusion_answer(
    vision_summary: str,
    context_records: List[dict],
    user_question: str,
    bookshelf_text: Optional[str] = None,
):
    blocked = safety_input_filter(vision_summary + " " + user_question)
    if blocked:
        return blocked

    guided_question = build_user_focused_question(user_question)

    prompt = build_fusion_prompt(
        vision_summary=vision_summary,
        context_records=context_records,
        user_question=guided_question,
        history=_history_text(),
    )

    raw = _generate(BASE_SYSTEM_PROMPT, prompt)
    safe = sanitize_output(raw)

    _append_history(user_question, safe)
    return safe


# =============================================================
# 4️⃣ PDF Explanation
# =============================================================
def generate_pdf_answer(report_text: str, user_question: Optional[str]) -> str:
    combined = (user_question or "") + " " + report_text[:500]

    blocked = safety_input_filter(combined)
    if blocked:
        return blocked

    prompt = build_pdf_prompt(report_text, user_question or "")
    raw = _generate(BASE_SYSTEM_PROMPT, prompt)
    return sanitize_output(raw)
