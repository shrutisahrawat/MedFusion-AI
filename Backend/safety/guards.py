# backend/safety/guards.py
import re

from .patterns import (
    EMERGENCY_PATTERNS,
    DOSAGE_PATTERNS,
    DIAGNOSIS_PATTERNS,
    SURGERY_PATTERNS,
    SENSITIVE_PATTERNS,
)

# ✅ Simple, consistent default warning (your requirement)
DEFAULT_SAFETY_MESSAGE = (
    "⚠️ This system is for assistance and knowledge purposes only.\n"
    "It is NOT a doctor and may make mistakes.\n"
    "Please consult a qualified doctor or medical professional for real medical advice."
)


def _match_any(text: str, patterns: list[str]) -> bool:
    lower = text.lower()
    return any(p in lower for p in patterns)


# -------- Individual checks --------

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

def redact_phi(text: str) -> str:
    """
    Basic privacy redaction for reports before sending to LLM.
    Not perfect, but reduces sharing of personal identifiers.
    """
    # Emails
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[REDACTED_EMAIL]", text)

    # Phone numbers (simple patterns, India + general)
    text = re.sub(r"\b(\+?\d[\d\-\s]{8,}\d)\b", "[REDACTED_PHONE]", text)

    # Dates (optional light redaction)
    text = re.sub(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", "[REDACTED_DATE]", text)

    # Common “Name:” / “Patient:” fields
    text = re.sub(r"(?i)\b(name|patient name)\s*:\s*[^\n]+", r"\1: [REDACTED_NAME]", text)
    text = re.sub(r"(?i)\b(patient|mr\.|mrs\.|ms\.)\s*[A-Za-z][A-Za-z\s]+", "[REDACTED_NAME]", text)

    return text
def trim_text(text: str, max_chars: int = 12000) -> tuple[str, bool]:
    """
    Keeps prompts within a safe size limit.
    Returns (trimmed_text, was_trimmed).
    """
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars] + "\n\n[TRUNCATED]", True


# -------- Main Input Guard --------

def safety_input_filter(user_text: str) -> str | None:
    """
    Returns a blocking safety message if the input is unsafe,
    otherwise returns None (safe to proceed).
    """

    if check_emergency(user_text):
        return (
            "🚨 This may indicate a medical emergency.\n"
            "Please seek immediate medical attention at the nearest hospital.\n\n"
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
            "Such cases always require direct medical supervision.\n\n"
            + DEFAULT_SAFETY_MESSAGE
        )

    # Diagnosis forcing is allowed but neutralized in output
    return None


# -------- Output Guard (Post-LLM) --------

def sanitize_output(text: str) -> str:
    """
    Ensures output stays non-diagnostic, non-prescriptive,
    and always appends the default safety warning.
    """
    lower = text.lower()

    # Stronger forbidden patterns (covers more ways LLM may diagnose)
    forbidden_phrases = [
    "this confirms you have",
    "you definitely have",
    "you are diagnosed with",
    "final diagnosis",
    "confirmed diagnosis",
    "start taking",
    "take medication",
    "take a medication",
    "take the medication",
    "increase the dose",
    "reduce the dose",
    "stop taking",
    "prescribe",
    "dosage",
]


    for phrase in forbidden_phrases:
        if phrase in lower:
            text = re.sub(
                re.escape(phrase),
                "this may suggest, but does NOT confirm (please consult a clinician)",
                text,
                flags=re.IGNORECASE,
            )
            lower = text.lower()

    # Always add your default safety note
    if DEFAULT_SAFETY_MESSAGE not in text:
        text += "\n\n" + DEFAULT_SAFETY_MESSAGE

    return text
