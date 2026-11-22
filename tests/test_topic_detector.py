"""Unit tests for topic detection, emergency triage, and chatbot routing."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from openai import OpenAIError

from app.chat.nodes import topic_detector
from app.chat.nodes.topic_detector import (
    EMERGENCY_DISCLAIMER,
    OFF_TOPIC_REFUSAL,
    detect_topic,
)
from app.chat.schemas import ChatMessage, GraphState
from app.llm import chatbot


@pytest.fixture
def mock_main_llm(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Fixture mocking the primary conversational LLM in chatbot."""
    fake = MagicMock()
    fake.invoke.return_value = AIMessage(content="LDL is a cholesterol carrier.")
    monkeypatch.setattr(chatbot, "llm", fake)
    return fake


@pytest.mark.parametrize(
    "query",
    [
        "I have severe chest pain and arm numbness",
        "I am short of breath right now",
        "My dad is having a stroke",
        "I suspect a heart attack",
        "Should I call 911?",
        "My mom's face is drooping",
        "His speech is slurred and confused",
        "I just fainted in the bathroom",
    ],
)
def test_emergency_keywords_fast_triage(
    monkeypatch: pytest.MonkeyPatch, query: str
) -> None:
    """Verify that acute symptom queries are triaged via deterministic keyword matching."""
    client_mock = MagicMock()
    monkeypatch.setattr(
        topic_detector, "_get_openai_client", lambda: (client_mock, "test-model")
    )

    state: GraphState = {"query": query, "history": []}
    result = detect_topic(state)

    assert result["topic"] == "emergency"
    assert result["status"] == "emergency"
    assert result["output_text"] == EMERGENCY_DISCLAIMER
    assert result["citations"] == []
    client_mock.chat.completions.create.assert_not_called()


@pytest.mark.parametrize(
    ("llm_response", "expected_topic", "expected_status", "expected_output"),
    [
        ("emergency", "emergency", "emergency", EMERGENCY_DISCLAIMER),
        ("off_topic", "off_topic", "off_topic", OFF_TOPIC_REFUSAL),
        ("cardiovascular_lifestyle", "cardiovascular_lifestyle", None, None),
    ],
)
def test_detect_topic_llm_classification(
    monkeypatch: pytest.MonkeyPatch,
    llm_response: str,
    expected_topic: str,
    expected_status: str | None,
    expected_output: str | None,
) -> None:
    """Verify LLM output parsing into emergency, off_topic, or lifestyle topics."""
    fake_msg = MagicMock()
    fake_msg.content = llm_response
    fake_msg.reasoning = None

    fake_choice = MagicMock()
    fake_choice.message = fake_msg

    fake_completion = MagicMock()
    fake_completion.choices = [fake_choice]

    client_mock = MagicMock()
    client_mock.chat.completions.create.return_value = fake_completion
    monkeypatch.setattr(
        topic_detector, "_get_openai_client", lambda: (client_mock, "test-model")
    )

    state: GraphState = {"query": "Some arbitrary health prompt", "history": []}
    result = detect_topic(state)

    assert result["topic"] == expected_topic
    if expected_status is not None:
        assert result.get("status") == expected_status
    if expected_output is not None:
        assert result.get("output_text") == expected_output


def test_detect_topic_fail_open_on_client_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify fail-open behavior to cardiovascular_lifestyle when LLM endpoint fails."""
    client_mock = MagicMock()
    client_mock.chat.completions.create.side_effect = OpenAIError("API unavailable")
    monkeypatch.setattr(
        topic_detector, "_get_openai_client", lambda: (client_mock, "test-model")
    )

    state: GraphState = {
        "query": "Is my blood pressure 135/85 borderline?",
        "history": [],
    }
    result = detect_topic(state)

    assert result["topic"] == "cardiovascular_lifestyle"


def test_detect_topic_includes_recent_history_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that recent user queries from history are included in the classifier prompt."""
    fake_msg = MagicMock()
    fake_msg.content = "cardiovascular_lifestyle"
    fake_choice = MagicMock()
    fake_choice.message = fake_msg
    fake_completion = MagicMock()
    fake_completion.choices = [fake_choice]

    client_mock = MagicMock()
    client_mock.chat.completions.create.return_value = fake_completion
    monkeypatch.setattr(
        topic_detector, "_get_openai_client", lambda: (client_mock, "test-model")
    )

    state: GraphState = {
        "query": "what about for post-menopausal women?",
        "history": [
            ChatMessage(role="user", content="What is the ideal LDL target?"),
            ChatMessage(role="assistant", content="The ideal LDL target is < 100."),
            ChatMessage(role="user", content="How does age affect this?"),
        ],
    }
    result = detect_topic(state)

    assert result["topic"] == "cardiovascular_lifestyle"
    call_args = client_mock.chat.completions.create.call_args
    sent_prompt = call_args.kwargs["messages"][0]["content"]
    assert "Prior context:" in sent_prompt
    assert "What is the ideal LDL target?" in sent_prompt
    assert "How does age affect this?" in sent_prompt


@pytest.mark.parametrize(
    ("topic", "reply"),
    [
        ("off_topic", OFF_TOPIC_REFUSAL),
        ("emergency", EMERGENCY_DISCLAIMER),
    ],
)
def test_chatbot_blocked_topics_skip_main_llm(
    monkeypatch: pytest.MonkeyPatch,
    mock_main_llm: MagicMock,
    topic: str,
    reply: str,
) -> None:
    """Verify that blocked topics bypass the main conversational LLM and return canned refusal."""
    monkeypatch.setattr(
        chatbot,
        "detect_topic",
        lambda _: {"topic": topic, "output_text": reply},
    )
    history = chatbot.new_history()

    response = chatbot.chat_with_bot("test message", history)
    assert response == reply
    mock_main_llm.invoke.assert_not_called()
    assert isinstance(history[-2], HumanMessage)
    assert history[-1].content == reply


def test_chatbot_on_topic_calls_main_llm(
    monkeypatch: pytest.MonkeyPatch,
    mock_main_llm: MagicMock,
) -> None:
    """Verify that on-topic lifestyle inquiries invoke the main LLM."""
    monkeypatch.setattr(
        chatbot,
        "detect_topic",
        lambda _: {"topic": "cardiovascular_lifestyle"},
    )
    history = chatbot.new_history()

    response = chatbot.chat_with_bot("What is LDL?", history)
    assert response == "LDL is a cholesterol carrier."
    mock_main_llm.invoke.assert_called_once()


def test_main_llm_gets_trimmed_history(
    monkeypatch: pytest.MonkeyPatch,
    mock_main_llm: MagicMock,
) -> None:
    """Verify that conversation history exceeding token budget is trimmed."""
    monkeypatch.setattr(
        chatbot,
        "detect_topic",
        lambda _: {"topic": "cardiovascular_lifestyle"},
    )
    history = chatbot.new_history()
    for i in range(50):
        history.extend(
            [
                HumanMessage(content=f"question {i} " * 20),
                AIMessage(content="answer " * 50),
            ]
        )

    chatbot.chat_with_bot("What is LDL?", history)

    sent = mock_main_llm.invoke.call_args.args[0]
    assert sent[0] is history[0]
    assert sent[-1].content == "What is LDL?"
    assert len(sent) < len(history)
