# backend/llm/llama_client.py
"""
Ollama-backed LLM client for MedFusion AI.

Talks to an Ollama HTTP server (default: http://localhost:11434/api/generate)
and uses prompt builders from backend/llm/prompt.py.

Safety:
- Runs safety_input_filter(...) before calling the LLM.
- Runs sanitize_output(...) on the raw LLM text before returning.
"""

import os
import requests
from typing import List, Optional

from backend.llm.prompt import (
    BASE_SYSTEM_PROMPT,
    build_text_rag_prompt,
    build_vision_prompt,
    build_pdf_prompt,
)
from backend.safety.guards import safety_input_filter, sanitize_output

# ---- Ollama endpoint configuration (change via env if needed) ----
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:latest")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "600"))  # seconds


# ---- Low-level Ollama caller ----
def _call_ollama(prompt: str, model: Optional[str] = None, timeout: Optional[int] = None) -> str:
    """
    Call Ollama HTTP API and return the assistant text.
    """
    _model = model or OLLAMA_MODEL
    _timeout = timeout or OLLAMA_TIMEOUT

    payload = {
        "model": _model,
        "prompt": prompt,
        "stream": False,
    }

    resp = requests.post(
        OLLAMA_URL,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=_timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    # Ollama may return a field called 'response' or similar depending on the server;
    # we attempt to read common keys.
    text = data.get("response") or data.get("text") or data.get("output") or ""
    return text.strip()


# ---- Helper to compose system + user prompt similar to the raw client example ----
def _compose_full_prompt(system_text: str, user_text: str) -> str:
    """
    Compose the final prompt in the form the Ollama endpoint expects.
    (Matches the example: system + newline + 'User: ...' + 'Assistant:')
    """
    return f"{system_text}\n\nUser: {user_text}\nAssistant:"


# ---- High-level API functions ----

def generate_text_rag_answer(context_chunks: List[str], user_question: str) -> str:
    """
    Generate an answer using PubMed RAG context + user question.
    Applies input safety check and output sanitization.
    """
    # Safety (input)
    blocked = safety_input_filter(user_question)
    if blocked is not None:
        return blocked

    # Build user prompt using prompt builder
    user_prompt = build_text_rag_prompt(context_chunks, user_question)

    # Compose full prompt with system text
    full_prompt = _compose_full_prompt(BASE_SYSTEM_PROMPT, user_prompt)

    # Call Ollama
    raw_text = _call_ollama(full_prompt)

    # Sanitize output and return
    safe_text = sanitize_output(raw_text)
    return safe_text


def generate_vision_answer(
    chest_summary: Optional[str],
    fracture_summary: Optional[str],
    user_description: Optional[str],
) -> str:
    """
    Generate an explanation from vision model outputs + optional user description.
    """
    combined_input = " ".join([t for t in [chest_summary, fracture_summary, user_description] if t])

    blocked = safety_input_filter(combined_input)
    if blocked is not None:
        return blocked

    user_prompt = build_vision_prompt(chest_summary, fracture_summary, user_description)
    full_prompt = _compose_full_prompt(BASE_SYSTEM_PROMPT, user_prompt)
    raw_text = _call_ollama(full_prompt)
    safe_text = sanitize_output(raw_text)
    return safe_text


def generate_pdf_answer(report_text: str, user_question: Optional[str]) -> str:
    """
    Explain a PDF-extracted medical report in simple language.
    """
    combined_input = (user_question or "") + " " + (report_text[:500] if report_text else "")

    blocked = safety_input_filter(combined_input)
    if blocked is not None:
        return blocked

    user_prompt = build_pdf_prompt(report_text, user_question or "")
    full_prompt = _compose_full_prompt(BASE_SYSTEM_PROMPT, user_prompt)
    raw_text = _call_ollama(full_prompt)
    safe_text = sanitize_output(raw_text)
    return safe_text
