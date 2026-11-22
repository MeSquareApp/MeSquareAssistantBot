"""Streamlit web interface for MeSquare Cardiovascular Assistant.

Integrates the compiled LangGraph clinical reasoning pipeline with interactive
digital twin parameter controls, guardrail status alerts, and guideline citations.
"""

from typing import Literal

import streamlit as st

from app.chat import app
from app.chat.schemas import (
    ChatMessage,
    CitationItem,
    DigitalTwinContext,
    GraphState,
    LabAnchors,
    RiskProfile,
    WearablesSummary,
)

st.set_page_config(page_title="MeSquare Assistant", layout="wide")
st.title("MeSquare Cardiovascular Assistant")
st.caption(
    "Evidence-based cardiovascular lifestyle, nutrition, and prevention guidance "
    "grounded in official clinical guidelines and patient digital twin biometrics."
)

# ---------------------------------------------------------
# Sidebar: Digital Twin Clinical Context Controls
# ---------------------------------------------------------
st.sidebar.header("Digital Twin Parameters")
st.sidebar.caption("Biomarkers and risk factors sent by MeSquareBE")

age = st.sidebar.number_input("Age", min_value=18, max_value=120, value=52)
sex_choice = st.sidebar.selectbox("Sex", options=["male", "female"], index=0)
sex_literal: Literal["male", "female"] = "male" if sex_choice == "male" else "female"
smoking = st.sidebar.selectbox(
    "Smoking Status", options=["non-smoker", "former smoker", "current smoker"], index=0
)

st.sidebar.subheader("Biomarkers & Labs")
ldl = st.sidebar.number_input(
    "LDL Cholesterol (mg/dL)", min_value=10.0, max_value=500.0, value=168.0
)
total_chol = st.sidebar.number_input(
    "Total Cholesterol (mg/dL)", min_value=50.0, max_value=600.0, value=245.0
)
hdl = st.sidebar.number_input(
    "HDL Cholesterol (mg/dL)", min_value=5.0, max_value=150.0, value=42.0
)
sys_bp = st.sidebar.number_input(
    "Systolic BP (mmHg)", min_value=70.0, max_value=250.0, value=138.0
)
dia_bp = st.sidebar.number_input(
    "Diastolic BP (mmHg)", min_value=40.0, max_value=150.0, value=88.0
)
prevent_score = st.sidebar.number_input(
    "AHA PREVENT 10-Yr Risk (%)", min_value=0.0, max_value=100.0, value=8.4
)

twin_context = DigitalTwinContext(
    user_id="streamlit_user_001",
    age=int(age),
    sex=sex_literal,
    smoking_status=str(smoking),
    lab_anchors=LabAnchors(
        total_cholesterol=float(total_chol),
        ldl=float(ldl),
        hdl=float(hdl),
        systolic_bp=float(sys_bp),
        diastolic_bp=float(dia_bp),
    ),
    wearables_summary=WearablesSummary(
        resting_heart_rate_avg_14d=74.0,
        step_count_avg_14d=4800.0,
        sleep_hours_avg_14d=6.2,
    ),
    risk_profile=RiskProfile(
        aha_prevent_10yr_score=float(prevent_score),
        nhanes_percentile=65.0,
        risk_category="borderline",
    ),
)

# ---------------------------------------------------------
# Chat Session State
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander(f"Official Guideline Citations ({len(msg['citations'])})"):
                for idx, c in enumerate(msg["citations"], start=1):
                    st.markdown(f"**[{idx}] {c['guideline_title']}**")
                    st.caption(f"Section: {c['section_title']}")
                    st.markdown(f"> {c['excerpt']}")

# ---------------------------------------------------------
# User Input & LangGraph Pipeline Execution
# ---------------------------------------------------------
prompt = st.chat_input("Ask about cardiovascular health, labs, or lifestyle...")

if prompt:
    # 1. Render user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Build history for LangGraph
    history_records: list[ChatMessage] = []
    for m in st.session_state.messages[:-1]:
        role_lit: Literal["user", "assistant", "system"] = (
            "user" if m["role"] == "user" else "assistant"
        )
        history_records.append(ChatMessage(role=role_lit, content=m["content"]))

    inputs: GraphState = {
        "query": prompt,
        "twin_context": twin_context,
        "history": history_records,
    }

    # 3. Execute StateGraph pipeline
    with st.chat_message("assistant"):
        with st.spinner("Processing through clinical reasoning graph..."):
            result = app.invoke(inputs)

        output_text = result.get("output_text", "")
        status = result.get("status", "error")
        citations: list[CitationItem] = result.get("citations", [])

        # Guardrail-specific display styling
        if status == "emergency":
            st.error(output_text)
        elif status == "medication_prohibited":
            st.warning(output_text)
        elif status == "off_topic":
            st.info(output_text)
        else:
            st.markdown(output_text)

        # Citations expander
        serialized_citations = []
        if citations:
            with st.expander(
                f"Official Guideline Citations ({len(citations)})", expanded=True
            ):
                for idx, c in enumerate(citations, start=1):
                    st.markdown(f"**[{idx}] {c.guideline_title}**")
                    st.caption(f"Section: {c.section_title}")
                    st.markdown(f"> {c.excerpt}")
                    serialized_citations.append(
                        {
                            "guideline_title": c.guideline_title,
                            "section_title": c.section_title,
                            "excerpt": c.excerpt,
                        }
                    )

    # 4. Save assistant response to session
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": output_text,
            "status": status,
            "citations": serialized_citations,
        }
    )
