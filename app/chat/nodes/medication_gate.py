"""Node 2: Deterministic pharmaceutical drug entity and intent gate.

Blocks clinical medication advice, prescriptions, and dosage adjustments
for common cardiovascular and lipid-lowering pharmacological agents.
"""

import re

from app.chat.schemas import GraphState

CARDIO_DRUGS: tuple[str, ...] = (
    # Statins
    "atorvastatin",
    "lipitor",
    "rosuvastatin",
    "crestor",
    "simvastatin",
    "zocor",
    "pravastatin",
    "lovastatin",
    # Blood pressure / ACE / ARB / Beta blockers
    "lisinopril",
    "losartan",
    "amlodipine",
    "metoprolol",
    "atenolol",
    "hydrochlorothiazide",
    "hctz",
    "valsartan",
    # Antiplatelets / Anticoagulants
    "plavix",
    "clopidogrel",
    "eliquis",
    "apixaban",
    "xarelto",
    "warfarin",
    "coumadin",
    # Other lipid lowering
    "ezetimibe",
    "zetia",
    "repatha",
    "evolocumab",
    "praluent",
    "bempedoic",
)

DRUG_PATTERN: re.Pattern[str] = re.compile(
    r"\b(" + "|".join(CARDIO_DRUGS) + r")\b", re.IGNORECASE
)

INTENT_PATTERNS: tuple[str, ...] = (
    r"\bshould i (take|start|stop|change|increase|decrease|switch)\b",
    r"\bcan i (take|increase|decrease|stop|start)\b",
    r"\bwhat (dose|dosage|medication|pill|drug)\b",
    r"\bprescribe\b",
)

MEDICATION_DISCLAIMER = (
    "Clinical Notice: As an AI assistant, I am not a physician and cannot "
    "recommend, modify, or prescribe medications or dosages. Please consult "
    "your physician or cardiologist regarding prescription therapies. I can, "
    "however, provide lifestyle, dietary, and habit adjustments based on "
    "official cardiovascular guidelines."
)


def check_medication(state: GraphState) -> dict[str, object]:
    """Inspect user query for pharmaceutical drug mentions or prescription intent."""
    query = state.get("query", "").lower()

    drug_match = DRUG_PATTERN.search(query)
    intent_match = any(re.search(p, query) for p in INTENT_PATTERNS)

    if drug_match or intent_match:
        return {
            "status": "medication_prohibited",
            "output_text": MEDICATION_DISCLAIMER,
            "citations": [],
        }

    return {"status": "success"}
