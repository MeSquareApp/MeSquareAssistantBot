import logging
from collections.abc import Sequence
from enum import StrEnum

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from app.core.config import settings
from app.llm.prompts import CLASSIFIER_PROMPT

logger = logging.getLogger(__name__)

# Earlier user messages sent to the classifier so short follow-ups keep their context.
# Assistant replies are left out: they are long and add little to the label.
CONTEXT_MESSAGES = 2


class TopicLabel(StrEnum):
    ON_TOPIC = "on_topic"
    OFF_TOPIC = "off_topic"
    EMERGENCY = "emergency"


class TopicDecision(BaseModel):
    label: TopicLabel


classifier_llm = ChatOpenAI(
    model=settings.classifier_model,
    api_key=SecretStr(settings.arc_llm_key),
    base_url=settings.arc_llm_endpoint,
    temperature=0,
    max_completion_tokens=100,
).with_structured_output(TopicDecision)


def classify(user_input: str, history: Sequence[BaseMessage]) -> TopicDecision:
    """Label the latest message; fail open so an outage never blocks health questions."""
    recent = [m for m in history if isinstance(m, HumanMessage)][-CONTEXT_MESSAGES:]
    messages = [
        SystemMessage(content=CLASSIFIER_PROMPT),
        *recent,
        HumanMessage(content=user_input),
    ]
    try:
        decision = classifier_llm.invoke(messages)
    except Exception:
        logger.exception("Topic classifier failed; treating message as on topic")
        return TopicDecision(label=TopicLabel.ON_TOPIC)
    if not isinstance(decision, TopicDecision):
        logger.warning("Topic classifier returned %r; treating as on topic", decision)
        return TopicDecision(label=TopicLabel.ON_TOPIC)
    logger.info("Topic: %s", decision.label)
    return decision
