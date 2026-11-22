# MeSquareAssistantBot

Health-education assistant service for biomarkers, lab results, and cardiovascular health. Built on FastAPI and LangGraph, this service integrates digital twin context, pgvector guideline retrieval, strict pharmaceutical and emergency guardrails, and real-time Server-Sent Events (SSE) streaming.

The assistant is not a medical doctor and does not diagnose, prescribe, or adjust medications.

---

## Requirements

- Python 3.15+
- [uv](https://docs.astral.sh/uv/)

---

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
ARC_LLM_CLASSIFIER_MODEL=   # model for topic triage; defaults to ARC_LLM_MODEL
ARC_LLM_CONTEXT_TOKENS=     # context window size, if model is not in CONTEXT_TOKENS
GENERATOR_MAX_TOKENS=1500   # maximum tokens generated per response
```

---

## Architectural Milestones

### Phase 1: Clinical Guideline RAG Knowledge Base

Establishes the vector knowledge base in Supabase PostgreSQL:
1. **Migrations**: `supabase db push` creates the `rag.guideline_chunks` table with `halfvec(2560)` and HNSW index.
2. **Ingestion**: `uv run python scripts/ingest_guidelines.py` processes PDFs via Docling, chunks by headings, generates embeddings, and upserts rows.
3. **Verification**: `uv run python scripts/verify_rag.py` checks latency, similarity thresholds, and text integrity.

### Phase 2: LangGraph Reasoning Pipeline & Clinical Guardrails

Assembles deterministic and LLM safety checks in `app/chat/graph.py`:
- **Topic Detector**: Fast emergency symptom triage and off-topic refusals.
- **Medication Gate**: Deterministic pharmaceutical regex blocking prescription requests.
- **Retriever**: Hybrid cosine pgvector search across AHA/ACC clinical guidelines.
- **Generator**: Twin-grounded clinical response synthesizer using `ChatOpenAI`.
- **Verifier**: Post-generation check preventing hallucinated drug names and building citations.

### Phase 3A: FastAPI SSE Streaming Service

Exposes high-throughput, non-blocking HTTP endpoints for MeSquare backend integration:
- `POST /internal/v1/chat/stream`: Streams execution events (`status`, `token`, `citations`, `done`, `error`) via Server-Sent Events (`text/event-stream`).
- `GET /health`: Liveness probe for load balancers and container orchestrators.
- `GET /health/db`: Database connectivity readiness probe.

---

## How to Run

### 1. FastAPI Backend API

Start the local development server (configured on port `8001` to avoid conflicting with services on `8000`):

```bash
uv run fastapi dev --port 8001
```

Or run directly with Uvicorn:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

For production deployment:

```bash
uv run fastapi run --port 8001
```

Interactive API documentation will be available at:
- **Swagger UI**: [http://localhost:8001/docs](http://localhost:8001/docs)
- **ReDoc**: [http://localhost:8001/redoc](http://localhost:8001/redoc)

---

### 2. Live SSE Streaming Request (curl)

Send a streaming chat request to verify Server-Sent Events in real time:

```bash
curl -N -X POST http://localhost:8001/internal/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "conv_123",
    "message_id": "msg_456",
    "query": "What dietary fiber helps lower LDL cholesterol?"
  }'
```

Example streaming response output:

```text
event: token
data: {"token": "Sol"}

event: token
data: {"token": "uble"}

event: token
data: {"token": " fiber"}

...

event: citations
data: {"citations": [{"chunk_id": "...", "guideline_title": "...", "section_title": "...", "excerpt": "..."}]}

event: done
data: {"status": "success"}
```

---

### 3. Alternative Interfaces

#### Web UI (Streamlit)
```bash
uv run streamlit run app/llm/streamlit_bot.py
```

#### Terminal Interactive Chat
```bash
uv run python -m app.llm.chatbot
```

---

## Testing & Quality Assurance

Run the test suite and quality gates:

```bash
# 1. Run all unit and integration tests
uv run pytest

# 2. Run only the FastAPI SSE streaming API test suite
uv run pytest tests/test_streaming_api.py -v

# 3. Static type checking (zero errors on strict mode)
uv run mypy --strict .

# 4. Code formatting and linting
uv run ruff check .
uv run ruff format --check .
```

---

## Project Layout

| Path | Purpose |
|---|---|
| `main.py` | Root FastAPI app re-export for standard ASGI servers |
| `app/main.py` | FastAPI application initialization, CORS, lifespan, and routing |
| `app/api/chat.py` | SSE streaming endpoint (`/internal/v1/chat/stream`) |
| `app/api/health.py` | Liveness (`/health`) and database readiness (`/health/db`) probes |
| `app/api/db_health.py` | Backward-compatible database health router |
| `app/core/db.py` | Async SQLAlchemy engine and session dependency |
| `app/core/config.py` | Pydantic settings and environment variable parser |
| `app/chat/graph.py` | LangGraph state graph wiring and node routing |
| `app/chat/schemas.py` | Pydantic data models (Digital Twin snapshot, ChatRequest, GraphState) |
| `app/chat/nodes/topic_detector.py` | Node 1: Emergency triage and topic classification |
| `app/chat/nodes/medication_gate.py` | Node 2: Pharmaceutical drug gate and prescription intent block |
| `app/chat/nodes/retriever.py` | Node 3: Hybrid pgvector guideline retriever |
| `app/chat/nodes/generator.py` | Node 4: Twin-grounded guideline response generator (`ChatOpenAI`) |
| `app/chat/nodes/verifier.py` | Node 5: Clinical safety validation and citation generator |
| `app/model/guideline.py` | GuidelineChunk SQLAlchemy ORM model with `halfvec(2560)` |
| `app/utils/pdf_chunk_handler.py` | Docling extraction, heading-based chunker, and embedding generator |
| `scripts/ingest_guidelines.py` | Ingestion CLI for processing clinical PDFs into Supabase |
| `scripts/verify_rag.py` | Database retrieval verification script |
| `tests/test_streaming_api.py` | Automated HTTP integration tests for SSE streaming and health checks |
| `tests/test_workflow.py` | LangGraph semantic pipeline and guardrail test suite |
| `tests/test_topic_detector.py` | Topic classification and emergency triage test suite |
