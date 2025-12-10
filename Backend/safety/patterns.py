# backend/safety/patterns.py

# -------- Emergency / Critical Symptoms --------
EMERGENCY_PATTERNS = [
    "severe chest pain",
    "crushing chest pain",
    "can't breathe",
    "cant breathe",
    "shortness of breath",
    "unconscious",
    "not breathing",
    "heavy bleeding",
    "vomiting blood",
    "seizure",
    "fits",
    "heart attack",
    "stroke",
    "paralysis",
]

# -------- Medicine & Dosage Misuse --------
DOSAGE_PATTERNS = [
    "how many mg",
    "how much mg",
    "dosage",
    "dose of",
    "how often should i take",
    "take paracetamol",
    "take ibuprofen",
    "take antibiotic",
    "which tablet",
    "which medicine should i take",
]

# -------- Diagnosis Forcing --------
DIAGNOSIS_PATTERNS = [
    "do i have",
    "do I have",
    "confirm that I have",
    "tell me if I have",
    "am i suffering from",
    "is it cancer",
    "is it serious disease",
]

# -------- Surgery & Invasive Treatment --------
SURGERY_PATTERNS = [
    "do i need surgery",
    "operation needed",
    "surgical treatment",
    "surgery risk",
]

# -------- Pregnancy & Child Treatment --------
SENSITIVE_PATTERNS = [
    "pregnant",
    "pregnancy",
    "newborn",
    "infant",
    "baby medicine",
    "child dosage",
]
