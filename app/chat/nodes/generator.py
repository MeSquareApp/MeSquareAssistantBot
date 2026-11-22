"""Node 4: Twin-grounded generation with official clinical guideline excerpts.

Constructs clinical context from patient digital twin parameters and retrieved
guidelines, dispatching to the configured reasoning model.
"""

import os

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.chat.schemas import (
    GraphState,
    LabAnchors,
    RetrievedChunk,
    RiskProfile,
    WearablesSummary,
)
from app.core.config import settings

SYSTEM_PROMPT = """You are MeSquare's Cardiovascular Lifestyle Assistant.
You provide evidence-based lifestyle, nutrition, and habit guidance for cardiovascular health.

CRITICAL CLINICAL RULES:
1. Base all guidance STRICTLY on the retrieved Guideline Excerpts below.
2. Personalize advice using the user's specific clinical metrics (labs, wearable averages, AHA PREVENT score).
3. NEVER prescribe, recommend, or suggest altering prescription medications.
4. For every recommendation, explicitly cite the guideline title and section.

USER DIGITAL TWIN SNAPSHOT:
{twin_str}

OFFICIAL GUIDELINE EXCERPTS:
{guidelines_str}
"""


def _get_chat_model() -> ChatOpenAI:
    """Resolve configured OpenAI-compatible ChatOpenAI model instance."""
    api_key = os.getenv("OPENAI_API_KEY") or settings.arc_llm_key
    base_url = os.getenv("OPENAI_BASE_URL") or settings.arc_llm_endpoint
    model = os.getenv("OPENAI_MODEL") or settings.arc_llm_model
    max_tokens = int(os.getenv("GENERATOR_MAX_TOKENS", "1500"))

    return ChatOpenAI(
        model=model,
        api_key=SecretStr(api_key),
        base_url=base_url,
        temperature=0.2,
        max_completion_tokens=max_tokens,
    )


def format_twin_context(state: GraphState) -> str:
    """Format digital twin metrics into structured clinical text."""
    ctx = state.get("twin_context")
    if not ctx:
        return "No personal digital twin metrics available."

    labs = ctx.lab_anchors or LabAnchors()
    wearables = ctx.wearables_summary or WearablesSummary()
    risk = ctx.risk_profile or RiskProfile()

    return (
        f"- Demographics: Age {ctx.age or 'Unknown'}, Sex {ctx.sex or 'Unknown'}, "
        f"Smoking: {ctx.smoking_status or 'Unknown'}\n"
        f"- AHA PREVENT 10-Year Total CVD Risk: {risk.aha_prevent_10yr_score or 'Not calculated'}% "
        f"(Percentile: {risk.nhanes_percentile or 'N/A'})\n"
        f"- Latest Labs: Total Cholesterol: {labs.total_cholesterol or 'N/A'} mg/dL, "
        f"LDL: {labs.ldl or 'N/A'} mg/dL, HDL: {labs.hdl or 'N/A'} mg/dL, "
        f"Blood Pressure: {labs.systolic_bp or 'N/A'}/{labs.diastolic_bp or 'N/A'} mmHg, "
        f"HbA1c: {labs.hba1c or 'N/A'}%\n"
        f"- 14-Day Wearable Averages: Resting Heart Rate: {wearables.resting_heart_rate_avg_14d or 'N/A'} bpm, "
        f"Daily Steps: {wearables.step_count_avg_14d or 'N/A'}, "
        f"Sleep: {wearables.sleep_hours_avg_14d or 'N/A'} hrs/night"
    )


def format_guidelines(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved guideline excerpts with chunk identifiers."""
    if not chunks:
        return "No specific guideline excerpts retrieved above similarity threshold."

    output: list[str] = []
    for c in chunks:
        output.append(
            f"[{c.chunk_id}] {c.guideline_title} ({c.section_title}):\n{c.content}\n"
        )
    return "\n---\n".join(output)


def generate_response(state: GraphState) -> dict[str, object]:
    """Generate personalized clinical lifestyle response grounded in guidelines."""
    twin_str = format_twin_context(state)
    chunks = state.get("retrieved_chunks", [])
    guidelines_str = format_guidelines(chunks)

    messages: list[BaseMessage] = [
        SystemMessage(
            content=SYSTEM_PROMPT.format(
                twin_str=twin_str,
                guidelines_str=guidelines_str,
            )
        )
    ]

    for msg in state.get("history", [])[-4:]:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            messages.append(AIMessage(content=msg.content))
        elif msg.role == "system":
            messages.append(SystemMessage(content=msg.content))

    messages.append(HumanMessage(content=state.get("query", "")))

    llm = _get_chat_model()
    response = llm.invoke(messages)

    content = str(response.content) if response.content else ""
    if not content and "reasoning" in response.additional_kwargs:
        content = str(response.additional_kwargs["reasoning"])

    return {"output_text": content}
