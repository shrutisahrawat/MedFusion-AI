# backend/safety/guards.py

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

    forbidden_phrases = [
        "this confirms you have",
        "you definitely have",
        "you are diagnosed with",
    ]

    for phrase in forbidden_phrases:
        if phrase in lower:
            text = text.replace(phrase, "this may suggest, but does NOT confirm")

    if DEFAULT_SAFETY_MESSAGE not in text:
        text += "\n\n" + DEFAULT_SAFETY_MESSAGE

    return text
