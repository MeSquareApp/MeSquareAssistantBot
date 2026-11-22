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
DATABASE_URL=postgresql+asyncpg://postgres.<project-ref>:<password>@aws-0-<region>.pooler.supabase.com:6543/postgres
ARC_LLM_ENDPOINT=https://llm-api.arc.vt.edu/api/v1
ARC_LLM_KEY=your-arc-api-key
ARC_LLM_MODEL=gpt-oss-120b
EMBEDDING_MODEL=Qwen3-Embedding-4B

# Optional
ARC_LLM_CLASSIFIER_MODEL=   # model for topic check; defaults to ARC_LLM_MODEL
ARC_LLM_CONTEXT_TOKENS=     # context window size, if model is not in CONTEXT_TOKENS
```

## Phase 1: Clinical Guideline RAG Knowledge Base

Phase 1 establishes the vector knowledge base for cardiovascular guidelines before building conversational agents in Phase 2.

### 1. Database Migrations

Apply database migrations to Supabase to create the `rag` schema, `rag.guideline_chunks` table with `halfvec(2560)`, and the HNSW cosine similarity index:

```bash
supabase db push
```

### 2. Ingest Guidelines

Place clinical guideline PDFs in `data/guidelines/`. Then run the ingestion script:

```bash
uv run python scripts/ingest_guidelines.py
```

The script runs a 4-step pipeline:
1. **Extract**: Converts PDF documents into Markdown using Docling.
2. **Chunk**: Slices Markdown into ~600-token chunks along heading boundaries.
3. **Embed**: Generates 2,560-dimensional embeddings using ARC VT's `Qwen3-Embedding-4B`.
4. **Load**: Upserts chunks and vectors into `rag.guideline_chunks` in Supabase Postgres.

Use `--dry-run` to test extraction and chunking without database writes:

```bash
uv run python scripts/ingest_guidelines.py --dry-run
```

### 3. Verify Knowledge Base (Step 9)

Run the verification script to prove that storage, embeddings, HNSW indexes, and queries meet all seven acceptance criteria:

```bash
uv run python scripts/verify_rag.py
```

The script checks:
- Zero connection errors to Supabase pooler on port 6543.
- Minimum of 3 retrieved chunks.
- Cosine similarity score >= 0.70 for top result.
- Clinical domain relevance matching dietary or lipid guidelines.
- Markdown text integrity with no binary garbage or empty content.
- Database query execution time under 25 ms.
- Clean exit code (code 0 on success, code 1 on failure).

## Run

FastAPI backend API:

```bash
uv run fastapi dev
```

Web UI (Streamlit):

```bash
uv run streamlit run app/llm/streamlit_bot.py
```

Terminal chat (type `exit` to quit):

```bash
uv run python -m app.llm.chatbot
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
| `app/main.py` | FastAPI application entrypoint |
| `app/api/db_health.py` | Database health check router |
| `app/core/db.py` | Async SQLAlchemy engine and session dependency |
| `app/core/config.py` | Pydantic settings and environment config |
| `app/model/guideline.py` | GuidelineChunk SQLAlchemy ORM model with `halfvec(2560)` |
| `app/utils/pdf_chunk_handler.py` | Docling extraction, token chunking, and embedding pipeline |
| `app/llm/prompts.py` | System prompt, classifier prompt and fixed replies |
| `app/chat/nodes/topic_detector.py` | Topic classifier and emergency triage node |
| `app/llm/chatbot.py` | Main model, history trimming, token report, terminal chat |
| `app/llm/streamlit_bot.py` | Streamlit web UI |
| `scripts/ingest_guidelines.py` | CLI orchestrator for guideline PDF ingestion |
| `scripts/verify_rag.py` | Step 9 RAG retrieval verification script |
