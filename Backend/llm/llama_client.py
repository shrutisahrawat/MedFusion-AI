# Backend/llm/llama_client.py

"""
LLaMA client wrapper for MedFusion AI using Ollama HTTP API.

- Uses local Ollama server (no GGUF / no llama-cpp required).
- Applies safety guards BEFORE and AFTER calling the model.
- Uses prompt builders from Backend/llm/prompt.py.
- Fully Python 3.9 compatible.
"""

import os
from typing import List, Optional

import requests

from Backend.llm.prompt import (
    BASE_SYSTEM_PROMPT,
    build_text_rag_prompt,
    build_vision_prompt,
    build_pdf_prompt,
)
from Backend.safety.guards import safety_input_filter, sanitize_output


# ---------- Ollama configuration ----------

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "512"))
TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))


def _generate(system_prompt: str, user_prompt: str) -> str:
    """
    Low-level text generation helper using Ollama.
    """
    full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
        },
    }

    resp = requests.post(
        OLLAMA_URL,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=600,
    )
    resp.raise_for_status()
    data = resp.json()

    # Different Ollama versions may use "response" or "text"
    text = data.get("response") or data.get("text") or ""
    return text.strip()


# ---------- High-level functions ----------

def generate_text_rag_answer(context_chunks: List[str], user_question: str) -> str:
    """
    Use PubMed RAG context + user question to generate an answer.
    Applies input filtering and output sanitization.
    """
    blocked = safety_input_filter(user_question)
    if blocked is not None:
        return blocked

    user_prompt = build_text_rag_prompt(context_chunks, user_question)
    raw = _generate(BASE_SYSTEM_PROMPT, user_prompt)
    safe = sanitize_output(raw)
    return safe


def generate_vision_answer(
    chest_summary: Optional[str],
    breast_summary: Optional[str],
    pneumonia_summary: Optional[str],
    organ_summary: Optional[str],
    user_description: Optional[str],
) -> str:
    """
    Use vision model summaries (ChestMNIST, BreastMNIST, PneumoniaMNIST, OrganAMNIST)
    plus optional user description to generate a cautious explanation.
    """
    combined_text = " ".join(
        [
            t
            for t in [
                chest_summary,
                breast_summary,
                pneumonia_summary,
                organ_summary,
                user_description,
            ]
            if t
        ]
    )

    blocked = safety_input_filter(combined_text)
    if blocked is not None:
        return blocked

    user_prompt = build_vision_prompt(
        chest_summary,
        breast_summary,
        pneumonia_summary,
        organ_summary,
        user_description,
    )
    raw = _generate(BASE_SYSTEM_PROMPT, user_prompt)
    safe = sanitize_output(raw)
    return safe


def generate_pdf_answer(report_text: str, user_question: Optional[str]) -> str:
    """
    Explain a medical report (PDF text) in simple language.
    """
    combined_input = (user_question or "") + " " + report_text[:500]

    blocked = safety_input_filter(combined_input)
    if blocked is not None:
        return blocked

    user_prompt = build_pdf_prompt(report_text, user_question or "")
    raw = _generate(BASE_SYSTEM_PROMPT, user_prompt)
    safe = sanitize_output(raw)
    return safe
