from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.core.config import settings
from app.llm.prompts import EMERGENCY_REPLY, OFF_TOPIC_REPLY, SYSTEM_PROMPT
from app.llm.topic_guard import TopicLabel, classify

llm = ChatOpenAI(
    model=settings.arc_llm_model,
    api_key=SecretStr(settings.arc_llm_key),
    base_url=settings.arc_llm_endpoint,
    temperature=0.7,
    max_completion_tokens=1000,
)

# Approximate token budget of conversation sent to the main model each turn.
# Older turns are dropped first; the system prompt is always kept.
HISTORY_TOKEN_BUDGET = 2000

Message = SystemMessage | HumanMessage | AIMessage

# Context windows published for the ARC-hosted models, in tokens.
CONTEXT_TOKENS = {
    "gpt-oss-120b": 131_072,
    "gpt-oss-120b-thinking-low": 131_072,
    "gpt-oss-120b-thinking-high": 131_072,
    "GLM-5.3": 131_072,
    "GLM-5.3-thinking-high": 131_072,
    "DeepSeek-V4.1-Flash": 1_048_576,
    "DeepSeek-V4.1-Flash-thinking-low": 1_048_576,
    "DeepSeek-V4.1-Flash-thinking-max": 1_048_576,
}

CANNED_REPLIES = {
    TopicLabel.OFF_TOPIC: OFF_TOPIC_REPLY,
    TopicLabel.EMERGENCY: EMERGENCY_REPLY,
}


def new_history() -> list[Message]:
    return [SystemMessage(content=SYSTEM_PROMPT)]


cli_history: list[Message] = new_history()


def context_limit() -> int | None:
    if settings.arc_llm_context_tokens:
        return settings.arc_llm_context_tokens
    return CONTEXT_TOKENS.get(settings.arc_llm_model)


def token_report(message: Message, history: list[Message]) -> str | None:
    """Tokens used by this reply, context remaining, and a running summary."""
    usage = message.usage_metadata if isinstance(message, AIMessage) else None
    if not usage:
        return None

    used = usage["total_tokens"]
    limit = context_limit()
    left = "unavailable" if limit is None else f"{max(limit - used, 0):,}"
    session_total = sum(
        item.usage_metadata["total_tokens"]
        for item in history
        if isinstance(item, AIMessage) and item.usage_metadata
    )
    summary = (
        f"prompt {usage['input_tokens']:,}, "
        f"completion {usage['output_tokens']:,}, "
        f"session {session_total:,}"
    )
    reasoning = (usage.get("output_token_details") or {}).get("reasoning")
    if reasoning:
        summary += f", reasoning {reasoning:,}"
    return f"Tokens used: {used:,}\nTokens left: {left}\nToken summary: {summary}"


def chat_with_bot(user_input: str, history: list[Message] | None = None) -> str:
    if history is None:
        history = cli_history
    decision = classify(user_input, history)
    history.append(HumanMessage(content=user_input))
    canned = CANNED_REPLIES.get(decision.label)
    if canned:
        history.append(AIMessage(content=canned))
        return canned
    context = trim_messages(
        history,
        max_tokens=HISTORY_TOKEN_BUDGET,
        token_counter=count_tokens_approximately,
        strategy="last",
        include_system=True,
        start_on="human",
    )
    response = llm.invoke(context)
    history.append(response)
    return response.text


if __name__ == "__main__":
    from app.chat import app
    from app.chat.schemas import GraphState

    print(
        "Welcome to MeSquare Cardiovascular Assistant! (LangGraph pipeline, type 'exit' to quit)"
    )
    while True:
        user_input = input("\nYou: ")
        if user_input.strip().lower() == "exit":
            break
        state: GraphState = {
            "query": user_input,
            "twin_context": None,
            "history": [],
        }
        result = app.invoke(state)
        status = result.get("status", "unknown")
        print(f"\n[{status.upper()}] Assistant:")
        print(result.get("output_text", ""))
        citations = result.get("citations", [])
        if citations:
            print(f"\nCitations ({len(citations)}):")
            for i, c in enumerate(citations, start=1):
                print(f"  [{i}] {c.guideline_title} ({c.section_title})")
