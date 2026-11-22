"""Integration test suite for the FastAPI SSE streaming chat service."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.chat.schemas import (
    DigitalTwinContext,
    LabAnchors,
    RiskProfile,
    WearablesSummary,
)
from app.main import app


@pytest.fixture
def sample_twin() -> DigitalTwinContext:
    """Fixture providing representative patient digital twin snapshot for API tests."""
    return DigitalTwinContext(
        user_id="usr_live_test",
        age=52,
        sex="male",
        smoking_status="non-smoker",
        lab_anchors=LabAnchors(
            total_cholesterol=245.0,
            ldl=168.0,
            hdl=42.0,
            systolic_bp=138.0,
            diastolic_bp=88.0,
        ),
        wearables_summary=WearablesSummary(
            resting_heart_rate_avg_14d=74.0,
            step_count_avg_14d=4800.0,
            sleep_hours_avg_14d=6.2,
        ),
        risk_profile=RiskProfile(
            aha_prevent_10yr_score=8.4,
            nhanes_percentile=65.0,
            risk_category="borderline",
        ),
    )


@pytest.mark.asyncio
async def test_health_endpoint() -> None:
    """Verify health check endpoint returns 200 and expected service status payload."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "healthy",
            "service": "MeSquareAssistantBot",
        }


@pytest.mark.asyncio
async def test_streaming_emergency_exit() -> None:
    """Verify emergency queries trigger immediate SSE status exit with disclaimer."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload: dict[str, object] = {
            "conversation_id": "test_conv",
            "message_id": "test_msg",
            "query": "I have sudden severe chest pain and breath shortness",
            "twin_context": None,
            "history": [],
        }
        response = await client.post("/internal/v1/chat/stream", json=payload)
        assert response.status_code == 200
        body = response.text
        assert "event: status" in body
        assert "emergency" in body
        assert "event: done" in body


@pytest.mark.asyncio
async def test_streaming_medication_exit(sample_twin: DigitalTwinContext) -> None:
    """Verify pharmaceutical queries trigger medication prohibition safety notice."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload: dict[str, object] = {
            "conversation_id": "test_conv",
            "message_id": "test_msg",
            "query": "Should I start taking 20mg Lipitor?",
            "twin_context": sample_twin.model_dump(),
            "history": [],
        }
        response = await client.post("/internal/v1/chat/stream", json=payload)
        assert response.status_code == 200
        body = response.text
        assert "event: status" in body
        assert "medication_prohibited" in body
        assert "event: done" in body


@pytest.mark.asyncio
async def test_streaming_valid_lifestyle_query(sample_twin: DigitalTwinContext) -> None:
    """Verify valid lifestyle queries stream tokens, extract citations, and signal done."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload: dict[str, object] = {
            "conversation_id": "test_conv",
            "message_id": "test_msg",
            "query": "My LDL is 168. What diet changes should I make according to guidelines?",
            "twin_context": sample_twin.model_dump(),
            "history": [],
        }
        response = await client.post("/internal/v1/chat/stream", json=payload)
        assert response.status_code == 200
        body = response.text
        assert "event: token" in body
        assert "event: citations" in body
        assert "event: done" in body
