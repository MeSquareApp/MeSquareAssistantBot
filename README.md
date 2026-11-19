# MeSquareAssistantBot

Health-education chatbot for biomarkers, lab results and cardiovascular health.
Each message is first checked by a topic classifier: off-topic requests get a
fixed refusal and possible emergencies get a "contact emergency services"
reply, without calling the main model. The assistant is not a doctor and does
not diagnose or prescribe.

## Requirements

- Python 3.15+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
```

Create a `.env` file in the project root:

```env
ARC_LLM_ENDPOINT=https://your-endpoint/v1
ARC_LLM_KEY=your-api-key
ARC_LLM_MODEL=gpt-oss-120b

# Optional
ARC_LLM_CLASSIFIER_MODEL=   # model for the topic check; defaults to ARC_LLM_MODEL
ARC_LLM_CONTEXT_TOKENS=     # context window size, if the model is not in CONTEXT_TOKENS
```

## Run

Web UI (Streamlit):

```bash
uv run streamlit run src/mesquareassistantbot/streamlit_bot.py
```

Terminal chat (type `exit` to quit):

```bash
uv run python -m mesquareassistantbot.chatbot
```

## Test

Unit tests (no API calls):

```bash
uv run pytest
```

Topic classifier accuracy against the live endpoint (run after changing
`prompts.py`):

```bash
uv run python tests/eval_topic_cases.py
```

## Project layout

| File | Purpose |
|---|---|
| `src/mesquareassistantbot/prompts.py` | System prompt, classifier prompt and fixed replies |
| `src/mesquareassistantbot/topic_guard.py` | Topic classifier (on topic / off topic / emergency) |
| `src/mesquareassistantbot/chatbot.py` | Main model, history trimming, token report, terminal chat |
| `src/mesquareassistantbot/streamlit_bot.py` | Streamlit web UI |
