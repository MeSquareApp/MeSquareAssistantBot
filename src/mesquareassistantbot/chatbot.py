import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_openai import ChatOpenAI

from mesquareassistantbot.prompts import EMERGENCY_REPLY, OFF_TOPIC_REPLY, SYSTEM_PROMPT
from mesquareassistantbot.topic_guard import TopicLabel, classify

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

llm = ChatOpenAI(
    model=os.environ["ARC_LLM_MODEL"],
    api_key=os.environ["ARC_LLM_KEY"],
    base_url=os.environ["ARC_LLM_ENDPOINT"],
    temperature=0.7,
    max_tokens=1000,
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
    override = os.environ.get("ARC_LLM_CONTEXT_TOKENS")
    if override:
        return int(override)
    return CONTEXT_TOKENS.get(os.environ["ARC_LLM_MODEL"])


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
    return (
        f"Tokens used: {used:,}\n"
        f"Tokens left: {left}\n"
        f"Token summary: {summary}"
    )


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
    print("Welcome to the AI Chatbot! (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        reply = chat_with_bot(user_input)
        print("Bot:", reply)
        report = token_report(cli_history[-1], cli_history)
        if report:
            print(report)
