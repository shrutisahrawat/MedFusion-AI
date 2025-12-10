# backend/llm/llama_client.py

"""
LLaMA client wrapper for MedFusion AI.

- Loads a local quantized LLaMA model via llama-cpp-python.
- Applies safety guards BEFORE and AFTER calling the model.
- Uses prompt builders from backend/llm/prompt.py.
"""

import os
from typing import List

from llama_cpp import Llama

from backend.llm.prompt import (
    BASE_SYSTEM_PROMPT,
    build_text_rag_prompt,
    build_vision_prompt,
    build_pdf_prompt,
)
from backend.safety.guards import safety_input_filter, sanitize_output


# ---------- Model configuration ----------

# You can set this via environment variable, or hard-code your .gguf path here
DEFAULT_MODEL_PATH = os.environ.get(
    "LLAMA_MODEL_PATH",
    "models/llama/llama-3.1-8b-instruct.Q4_K_M.gguf",  # <- adjust to your folder
)

MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "512"))
TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))

# Global singleton instance
_llm: Llama | None = None


def get_llm() -> Llama:
    """
    Lazy-load a single LLaMA model instance.
    """
    global _llm
    if _llm is None:
        if not os.path.exists(DEFAULT_MODEL_PATH):
            raise FileNotFoundError(
                f"LLaMA model file not found at: {DEFAULT_MODEL_PATH}\n"
                "Please download a quantized .gguf model and update LLAMA_MODEL_PATH."
            )

        _llm = Llama(
            model_path=DEFAULT_MODEL_PATH,
            n_ctx=4096,
            logits_all=False,
            n_threads=4,   # adjust based on your CPU cores
            use_mlock=False,
            verbose=False,
        )
    return _llm


def _generate(system_prompt: str, user_prompt: str) -> str:
    """
    Low-level text generation helper.
    """
    llm = get_llm()
    prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
    output = llm(
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stop=["<|user|>", "<|system|>", "</s>"],
    )
    text = output["choices"][0]["text"]
    return text.strip()


# ---------- High-level functions ----------

def generate_text_rag_answer(context_chunks: List[str], user_question: str) -> str:
    """
    Use PubMed RAG context + user question to generate an answer.
    Applies input filtering and output sanitization.
    """
    # Input safety
    blocked = safety_input_filter(user_question)
    if blocked is not None:
        return blocked

    user_prompt = build_text_rag_prompt(context_chunks, user_question)
    raw = _generate(BASE_SYSTEM_PROMPT, user_prompt)
    safe = sanitize_output(raw)
    return safe


def generate_vision_answer(
    chest_summary: str | None,
    fracture_summary: str | None,
    user_description: str | None,
) -> str:
    """
    Use vision model summaries + optional user description to generate explanation.
    """
    combined_text = " ".join(
        [t for t in [chest_summary, fracture_summary, user_description] if t]
    )

    blocked = safety_input_filter(combined_text)
    if blocked is not None:
        return blocked

    user_prompt = build_vision_prompt(chest_summary, fracture_summary, user_description)
    raw = _generate(BASE_SYSTEM_PROMPT, user_prompt)
    safe = sanitize_output(raw)
    return safe


def generate_pdf_answer(report_text: str, user_question: str | None) -> str:
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
