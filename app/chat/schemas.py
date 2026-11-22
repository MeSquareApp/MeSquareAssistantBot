"""Domain schemas and graph state contracts for MeSquareAssistantBot.

Defines the Digital Twin context contracts, clinical message schemas,
retrieved guideline chunk models, citation structures, and LangGraph state.
"""

from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field


class LabAnchors(BaseModel):
    """Laboratory biomarker anchors from the patient health record."""

    model_config = ConfigDict(extra="ignore")

    total_cholesterol: float | None = None
    hdl: float | None = None
    ldl: float | None = None
    systolic_bp: float | None = None
    diastolic_bp: float | None = None
    fasting_glucose: float | None = None
    hba1c: float | None = None
    anchor_date: str | None = None


class WearablesSummary(BaseModel):
    """14-day aggregated wearable biometrics."""

    model_config = ConfigDict(extra="ignore")

    resting_heart_rate_avg_14d: float | None = None
    step_count_avg_14d: float | None = None
    sleep_hours_avg_14d: float | None = None


class RiskProfile(BaseModel):
    """Cardiovascular risk stratification and population percentiles."""

    model_config = ConfigDict(extra="ignore")

    aha_prevent_10yr_score: float | None = None
    nhanes_percentile: float | None = None
    risk_category: str | None = None


class DigitalTwinContext(BaseModel):
    """Personalized clinical snapshot sent by MeSquareBE."""

    model_config = ConfigDict(extra="ignore")

    user_id: str
    age: int | None = None
    sex: Literal["male", "female"] | None = None
    smoking_status: str | None = None
    lab_anchors: LabAnchors | None = Field(default_factory=LabAnchors)
    wearables_summary: WearablesSummary | None = Field(default_factory=WearablesSummary)
    risk_profile: RiskProfile | None = Field(default_factory=RiskProfile)


class CitationItem(BaseModel):
    """Formal citation item referencing official guideline evidence."""

    model_config = ConfigDict(extra="ignore")

    chunk_id: str
    guideline_title: str
    section_title: str
    excerpt: str


class ChatMessage(BaseModel):
    """Individual chat message within conversation history."""

    model_config = ConfigDict(extra="ignore")

    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    """Inbound request payload for clinical conversation endpoints."""

    model_config = ConfigDict(extra="ignore")

    conversation_id: str
    message_id: str
    query: str
    twin_context: DigitalTwinContext | None = None
    history: list[ChatMessage] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    """Guideline chunk retrieved from pgvector similarity search."""

    model_config = ConfigDict(extra="ignore")

    chunk_id: str
    guideline_title: str
    section_title: str
    content: str
    similarity: float


class GraphState(TypedDict, total=False):
    """Internal state tracked across nodes in the LangGraph semantic pipeline."""

    query: str
    twin_context: DigitalTwinContext | None
    history: list[ChatMessage]
    topic: Literal["cardiovascular_lifestyle", "off_topic", "emergency"]
    status: Literal[
        "success", "off_topic", "emergency", "medication_prohibited", "error"
    ]
    retrieved_chunks: list[RetrievedChunk]
    output_text: str
    citations: list[CitationItem]
