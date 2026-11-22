"""Chat API endpoints for agentic reasoning and SSE streaming.

Wraps the compiled LangGraph clinical reasoning workflow and streams execution
events (early exit notices, generator token chunks, citations, and completion
markers) as Server-Sent Events (SSE).
"""

import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, status
from openai import OpenAIError
from sqlalchemy.exc import SQLAlchemyError
from sse_starlette.sse import EventSourceResponse

from app.chat.graph import app as workflow_app
from app.chat.schemas import ChatRequest, GraphState

router = APIRouter(tags=["chat"])


def _extract_citations(raw_citations: object) -> list[dict[str, object]]:
    """Convert raw citation models or dictionaries into JSON-serializable structures."""
    citations: list[dict[str, object]] = []
    if isinstance(raw_citations, list):
        for item in raw_citations:
            if hasattr(item, "model_dump"):
                citations.append(item.model_dump())
            elif isinstance(item, dict):
                citations.append(item)
    return citations


async def stream_graph_events(
    request: ChatRequest,
) -> AsyncGenerator[dict[str, str]]:
    """Stream LangGraph execution events as Server-Sent Events (SSE)."""
    initial_state: GraphState = {
        "query": request.query,
        "twin_context": request.twin_context,
        "history": request.history,
        "topic": "cardiovascular_lifestyle",
        "status": "success",
        "retrieved_chunks": [],
        "output_text": "",
        "citations": [],
    }

    try:
        async for event in workflow_app.astream_events(initial_state, version="v2"):
            kind = event.get("event")
            metadata = event.get("metadata", {})
            node_name = (
                metadata.get("langgraph_node", "") if isinstance(metadata, dict) else ""
            )

            # 1. Early exit node completions (Topic triage & medication blocks)
            if kind == "on_chain_end" and node_name in (
                "topic_detector",
                "medication_gate",
            ):
                event_data = event.get("data")
                raw_output = (
                    event_data.get("output") if isinstance(event_data, dict) else None
                )
                data: dict[str, object] = (
                    raw_output if isinstance(raw_output, dict) else {}
                )
                status_val = data.get("status")
                if status_val in (
                    "emergency",
                    "off_topic",
                    "medication_prohibited",
                ):
                    yield {
                        "event": "status",
                        "data": json.dumps(
                            {
                                "status": status_val,
                                "output_text": data.get("output_text"),
                            }
                        ),
                    }
                    yield {
                        "event": "done",
                        "data": json.dumps(
                            {"status": status_val, "total_citations": 0}
                        ),
                    }
                    return

            # 2. Model token streaming from the generator node
            elif kind == "on_chat_model_stream" and node_name == "generator":
                event_data = event.get("data")
                chunk = (
                    event_data.get("chunk") if isinstance(event_data, dict) else None
                )
                content = getattr(chunk, "content", None)
                if content:
                    yield {
                        "event": "token",
                        "data": json.dumps({"token": content}),
                    }

            # 3. Final verification and citation emission
            elif kind == "on_chain_end" and node_name == "verifier":
                event_data = event.get("data")
                raw_output = (
                    event_data.get("output") if isinstance(event_data, dict) else None
                )
                data = raw_output if isinstance(raw_output, dict) else {}
                status_val = data.get("status")
                if status_val == "medication_prohibited":
                    yield {
                        "event": "status",
                        "data": json.dumps(
                            {
                                "status": status_val,
                                "output_text": data.get("output_text"),
                            }
                        ),
                    }
                    yield {
                        "event": "done",
                        "data": json.dumps(
                            {"status": status_val, "total_citations": 0}
                        ),
                    }
                    return

                citations = _extract_citations(data.get("citations"))
                yield {
                    "event": "citations",
                    "data": json.dumps({"citations": citations}),
                }

        # 4. Final completion event
        yield {
            "event": "done",
            "data": json.dumps({"status": "success"}),
        }

    except (
        OpenAIError,
        SQLAlchemyError,
        RuntimeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as exc:
        yield {
            "event": "error",
            "data": json.dumps({"error": str(exc)}),
        }


@router.post(
    "/internal/v1/chat/stream",
    status_code=status.HTTP_200_OK,
    summary="Stream conversational clinical RAG tokens and citations",
)
async def chat_stream(request: ChatRequest) -> EventSourceResponse:
    """Stream LangGraph clinical workflow execution events as Server-Sent Events."""
    return EventSourceResponse(stream_graph_events(request))
