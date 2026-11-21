from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from mesquareassistantbot import chatbot, topic_guard
from mesquareassistantbot.prompts import EMERGENCY_REPLY, OFF_TOPIC_REPLY
from mesquareassistantbot.topic_guard import TopicDecision, TopicLabel


@pytest.fixture
def main_llm(monkeypatch):
    fake = MagicMock()
    fake.invoke.return_value = AIMessage(content="LDL is a cholesterol carrier.")
    monkeypatch.setattr(chatbot, "llm", fake)
    return fake


def set_label(monkeypatch, label: TopicLabel) -> None:
    decision = TopicDecision(label=label)
    monkeypatch.setattr(chatbot, "classify", lambda *_: decision)


@pytest.mark.parametrize(
    ("label", "reply"),
    [(TopicLabel.OFF_TOPIC, OFF_TOPIC_REPLY), (TopicLabel.EMERGENCY, EMERGENCY_REPLY)],
)
def test_blocked_labels_skip_main_llm(monkeypatch, main_llm, label, reply):
    set_label(monkeypatch, label)
    history = chatbot.new_history()

    assert chatbot.chat_with_bot("anything", history) == reply
    main_llm.invoke.assert_not_called()
    assert isinstance(history[-2], HumanMessage)
    assert history[-1].content == reply


def test_on_topic_calls_main_llm(monkeypatch, main_llm):
    set_label(monkeypatch, TopicLabel.ON_TOPIC)
    history = chatbot.new_history()

    assert chatbot.chat_with_bot("What is LDL?", history) == "LDL is a cholesterol carrier."
    main_llm.invoke.assert_called_once()


def test_classifier_failure_fails_open(monkeypatch, main_llm):
    broken = MagicMock()
    broken.invoke.side_effect = RuntimeError("endpoint down")
    monkeypatch.setattr(topic_guard, "classifier_llm", broken)
    monkeypatch.setattr(chatbot, "classify", topic_guard.classify)

    chatbot.chat_with_bot("What is LDL?", chatbot.new_history())
    main_llm.invoke.assert_called_once()


def test_main_llm_gets_trimmed_history(monkeypatch, main_llm):
    set_label(monkeypatch, TopicLabel.ON_TOPIC)
    history = chatbot.new_history()
    for i in range(50):
        history += [HumanMessage(content=f"question {i} " * 20), AIMessage(content="answer " * 50)]

    chatbot.chat_with_bot("What is LDL?", history)

    sent = main_llm.invoke.call_args.args[0]
    assert sent[0] is history[0]
    assert sent[-1].content == "What is LDL?"
    assert len(sent) < len(history)


def test_classifier_receives_recent_context_only(monkeypatch):
    spy = MagicMock()
    spy.invoke.return_value = TopicDecision(label=TopicLabel.ON_TOPIC)
    monkeypatch.setattr(topic_guard, "classifier_llm", spy)
    history = chatbot.new_history()
    for i in range(5):
        history += [HumanMessage(content=f"q{i}"), AIMessage(content=f"a{i}")]

    topic_guard.classify("what about for women?", history)

    sent = spy.invoke.call_args.args[0]
    # classifier prompt + last 2 user messages + new message; replies and main prompt excluded
    assert [m.content for m in sent[1:]] == ["q3", "q4", "what about for women?"]
