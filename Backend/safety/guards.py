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
# Default Safety Message (same everywhere in system)
# ------------------------------------------------------------
DEFAULT_SAFETY_MESSAGE = (
    "⚠️ This system is for assistance and knowledge purposes only.\n"
    "It is NOT a doctor and may make mistakes.\n"
    "Please consult a qualified doctor or medical professional for real medical advice."
)

# ------------------------------------------------------------
# Generic pattern matching
# ------------------------------------------------------------
def _match_any(text: str, patterns: List[str]) -> bool:
    lower = text.lower()
    return any(p in lower for p in patterns)

# ------------------------------------------------------------
# Individual checks
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
# PHI Redaction (PDF Safety)
# ------------------------------------------------------------
def redact_phi(text: str) -> str:
    """Redacts emails, phone numbers, dates, and names from medical reports."""
    # Emails
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
# Text trimming for PDF → LLM
# ------------------------------------------------------------
def trim_text(text: str, max_chars: int = 12000):
    """Returns (trimmed_text, was_trimmed_boolean)."""
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars] + "\n\n[TRUNCATED]", True

# ------------------------------------------------------------
# INPUT SAFETY FILTER
# ------------------------------------------------------------
def safety_input_filter(user_text: str) -> Optional[str]:
    """Returns an error message if unsafe; otherwise None."""

    if check_emergency(user_text):
        return (
            "🚨 This may indicate a medical emergency.\n"
            "Please seek immediate medical attention.\n\n"
            + DEFAULT_SAFETY_MESSAGE
        )

    if check_dosage(user_text):
        return (
            "⚠️ I cannot provide medicine names, dosages, or prescriptions.\n\n"
            + DEFAULT_SAFETY_MESSAGE
        )

    if check_surgery(user_text):
        return (
            "⚠️ I cannot advise on surgical decisions or procedures.\n\n"
            + DEFAULT_SAFETY_MESSAGE
        )

    if check_sensitive_case(user_text):
        return (
            "⚠️ This query involves pregnancy, infants, or children.\n"
            "Such cases require direct medical supervision.\n\n"
            + DEFAULT_SAFETY_MESSAGE
        )

    # Diagnosis requests allowed, but LLM output safely neutralized
    return None

# ------------------------------------------------------------
# OUTPUT SANITIZATION (VERY IMPORTANT)
# ------------------------------------------------------------
def sanitize_output(text: str) -> str:
    """Prevents LLM from making diagnoses or giving prescriptions."""

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
                "this may suggest but does NOT confirm (consult a clinician)",
                text,
                flags=re.IGNORECASE,
            )
            lower = text.lower()

    # Always add safety message
    if DEFAULT_SAFETY_MESSAGE not in text:
        text += "\n\n" + DEFAULT_SAFETY_MESSAGE

    return text
