"""Node 1: Topic detection and emergency triage classifier.

Classifies inbound patient inquiries into cardiovascular lifestyle questions,
acute emergency symptoms, or out-of-scope/off-topic requests.
"""

import os

from openai import OpenAI, OpenAIError

from app.core.config import settings
from schemas import GraphState

TOPIC_PROMPT = """You are a medical triage classifier for a cardiovascular digital twin app.
Classify the user inquiry into exactly one category:
1. 'emergency': User mentions acute symptoms like chest pain, pressure, shortness of breath, numbness in arm, fainting.
2. 'cardiovascular_lifestyle': User asks about heart health, diet, cholesterol, exercise, blood pressure, labs, or habits.
3. 'off_topic': General chit-chat, coding, finance, sports, or unrelated medical questions (e.g. broken bone).

Respond with only one word: 'emergency', 'cardiovascular_lifestyle', or 'off_topic'.
Query: {query}"""

EMERGENCY_DISCLAIMER = (
    "EMERGENCY NOTICE: If you are experiencing chest pain, pressure, shortness of "
    "breath, or stroke symptoms, please call 911 or visit the nearest emergency "
    "room immediately."
)

OFF_TOPIC_REFUSAL = (
    "I am MeSquare's Cardiovascular Lifestyle Assistant. I can only assist with "
    "cardiovascular health, habit adjustments, lab interpretations, and lifestyle "
    "prevention guidelines."
)

EMERGENCY_KEYWORDS: tuple[str, ...] = (
    "chest pain",
    "short of breath",
    "stroke",
    "heart attack",
    "call 911",
)


def _get_openai_client() -> tuple[OpenAI, str]:
    """Resolve OpenAI-compatible client and effective model name."""
    api_key = os.getenv("OPENAI_API_KEY") or settings.arc_llm_key
    base_url = os.getenv("OPENAI_BASE_URL") or settings.arc_llm_endpoint
    client = OpenAI(api_key=api_key, base_url=base_url)
    model = (
        os.getenv("CLASSIFIER_MODEL")
        or settings.classifier_model
        or settings.arc_llm_model
    )
    return client, model


def detect_topic(state: GraphState) -> dict[str, object]:
    """Classify user query into emergency, off_topic, or cardiovascular_lifestyle."""
    query = state.get("query", "")
    lower_q = query.lower()

    if any(k in lower_q for k in EMERGENCY_KEYWORDS):
        return {
            "topic": "emergency",
            "status": "emergency",
            "output_text": EMERGENCY_DISCLAIMER,
            "citations": [],
        }

    client, model = _get_openai_client()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": TOPIC_PROMPT.format(query=query)}],
            temperature=0.0,
            max_tokens=500,
        )
        msg = response.choices[0].message
        reasoning = getattr(msg, "reasoning", "")
        raw_content = msg.content or (str(reasoning) if reasoning else "")
        classified = raw_content.strip().lower()
    except OpenAIError, RuntimeError, ValueError:
        # Fallback to cardiovascular lifestyle if LLM is unavailable
        classified = "cardiovascular_lifestyle"

    if "emergency" in classified:
        return {
            "topic": "emergency",
            "status": "emergency",
            "output_text": EMERGENCY_DISCLAIMER,
            "citations": [],
        }

    if "off_topic" in classified:
        return {
            "topic": "off_topic",
            "status": "off_topic",
            "output_text": OFF_TOPIC_REFUSAL,
            "citations": [],
        }

    return {"topic": "cardiovascular_lifestyle"}
