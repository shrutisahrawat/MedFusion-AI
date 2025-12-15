# Backend/safety/guards.py

import re
from typing import Optional, List

from .patterns import (
    EMERGENCY_PATTERNS,
    DOSAGE_PATTERNS,
    DIAGNOSIS_PATTERNS,
    SURGERY_PATTERNS,
    SENSITIVE_PATTERNS,
)

# ------------------------------------------------------------
# Default Safety Message (used across entire project)
# ------------------------------------------------------------
DEFAULT_SAFETY_MESSAGE = (
    "⚠️ This system is for assistance and educational purposes only.\n"
    "It is NOT a doctor and may make mistakes.\n"
    "Always consult a qualified medical professional for real medical advice."
)

# ------------------------------------------------------------
# Generic pattern matcher
# ------------------------------------------------------------
def _match_any(text: str, patterns: List[str]) -> bool:
    lower = text.lower()
    return any(p in lower for p in patterns)

# ------------------------------------------------------------
# Individual safety checks
# ------------------------------------------------------------
def check_emergency(text: str) -> bool:
    return _match_any(text, EMERGENCY_PATTERNS)

def check_dosage(text: str) -> bool:
    return _match_any(text, DOSAGE_PATTERNS)

def check_diagnosis_request(text: str) -> bool:
    return _match_any(text, DIAGNOSIS_PATTERNS)

def check_surgery(text: str) -> bool:
    return _match_any(text, SURGERY_PATTERNS)

def check_sensitive_case(text: str) -> bool:
    return _match_any(text, SENSITIVE_PATTERNS)

# ------------------------------------------------------------
# PHI REDACTION (PDF safety)
# ------------------------------------------------------------
def redact_phi(text: str) -> str:
    """Redacts personal identifiers before sending report to LLM."""

    # Email addresses
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[REDACTED_EMAIL]", text)

    # Phone numbers
    text = re.sub(r"\b(\+?\d[\d\-\s]{8,}\d)\b", "[REDACTED_PHONE]", text)

    # Dates
    text = re.sub(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", "[REDACTED_DATE]", text)

    # Names
    text = re.sub(r"(?i)(patient name|name)\s*:\s*[^\n]+", r"\1: [REDACTED_NAME]", text)
    text = re.sub(r"(?i)\b(mr\.|mrs\.|ms\.)\s+[A-Za-z][A-Za-z\s]+", "[REDACTED_NAME]", text)

    return text

# ------------------------------------------------------------
# TEXT TRIMMING
# ------------------------------------------------------------
def trim_text(text: str, max_chars: int = 12000):
    """Trims medical report for LLM token safety."""
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars] + "\n\n[TRUNCATED]", True

# ------------------------------------------------------------
# INPUT SAFETY FILTER
# ------------------------------------------------------------
def safety_input_filter(user_text: str) -> Optional[str]:
    """
    Returns a blocking safety message if unsafe.
    Otherwise returns None.
    """

    if check_emergency(user_text):
        return (
            "🚨 This appears to describe a potential medical emergency.\n"
            "Please seek immediate medical attention.\n\n"
            + DEFAULT_SAFETY_MESSAGE
        )

    if check_dosage(user_text):
        return (
            "⚠️ I cannot provide medicine names, dosages, or prescription guidance.\n\n"
            + DEFAULT_SAFETY_MESSAGE
        )

    if check_surgery(user_text):
        return (
            "⚠️ I cannot provide guidance on surgical procedures or decisions.\n\n"
            + DEFAULT_SAFETY_MESSAGE
        )

    if check_sensitive_case(user_text):
        return (
            "⚠️ This involves pregnancy or children — these cases require direct medical supervision.\n\n"
            + DEFAULT_SAFETY_MESSAGE
        )

    # Diagnosis questions allowed but will be neutralized in output
    return None

# ------------------------------------------------------------
# OUTPUT SANITIZATION (Post-LLM Safety)
# ------------------------------------------------------------
def sanitize_output(text: str) -> str:
    """
    Removes prompt leakage, fixes formatting,
    neutralizes unsafe claims, and appends disclaimer once.
    """

    if not text:
        return text

    # ====================================================
    # 🧹 ADDITION 1: REMOVE PROMPT / INSTRUCTION LEAKS
    # ====================================================
    leak_patterns = [
        r"=+\s*EXPLANATION REQUIREMENTS.*?\n",
        r"=+\s*PUBMED EVIDENCE.*?\n",
        r"=+\s*DISCLAIMER.*?\n",
        r"EXPLANATION REQUIREMENTS",
        r"PUBMED EVIDENCE\s*\(USE THIS\)",
        r"DISCLAIMER\s*={2,}.*",
    ]

    for pattern in leak_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    # ====================================================
    # 🔒 YOUR EXISTING SAFETY LOGIC (UNCHANGED)
    # ====================================================
    forbidden_phrases = [
        "this confirms you have",
        "you definitely have",
        "you are diagnosed with",
        "confirmed diagnosis",
        "final diagnosis",
        "start taking",
        "take medication",
        "take the medication",
        "increase the dose",
        "reduce the dose",
        "prescribe",
        "dosage",
    ]

    lower = text.lower()

    for phrase in forbidden_phrases:
        if phrase in lower:
            text = re.sub(
                re.escape(phrase),
                "this may suggest something, but does NOT confirm anything (please consult a clinician)",
                text,
                flags=re.IGNORECASE,
            )
            lower = text.lower()

    # ====================================================
    # 🧹 ADDITION 2: REMOVE DUPLICATE DISCLAIMERS
    # ====================================================
    disclaimer_regex = re.escape(DEFAULT_SAFETY_MESSAGE)
    text = re.sub(
        f"({disclaimer_regex}\\s*)+",
        DEFAULT_SAFETY_MESSAGE + "\n",
        text,
        flags=re.IGNORECASE,
    )

    # ====================================================
    # 🧹 ADDITION 3: FORMAT CLEANUP
    # ====================================================
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # ====================================================
    # 🔚 YOUR EXISTING DISCLAIMER GUARANTEE
    # ====================================================
    if DEFAULT_SAFETY_MESSAGE not in text:
        text = text.rstrip() + "\n\n" + DEFAULT_SAFETY_MESSAGE

    return text
