"""Compiled LangGraph semantic pipeline and guardrails for MeSquareAssistantBot.

Assembles topic detection, pharmaceutical gating, pgvector retrieval,
twin-grounded generation, and citation verification into a compiled state graph.
"""

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from nodes.generator import generate_response
from nodes.medication_gate import check_medication
from nodes.retriever import retrieve_guidelines
from nodes.topic_detector import detect_topic
from nodes.verifier import verify_output
from schemas import GraphState

GraphApp = CompiledStateGraph[GraphState, None, GraphState, GraphState]


def route_after_topic(state: GraphState) -> str:
    """Route inquiry after topic detection stage."""
    topic = state.get("topic")
    if topic in ("emergency", "off_topic"):
        return END
    return "medication_gate"


def route_after_medication(state: GraphState) -> str:
    """Route inquiry after pharmaceutical gate validation."""
    if state.get("status") == "medication_prohibited":
        return END
    return "retriever"


def build_workflow() -> GraphApp:
    """Construct and compile the clinical reasoning StateGraph."""
    workflow = StateGraph(GraphState)

    workflow.add_node("topic_detector", detect_topic)
    workflow.add_node("medication_gate", check_medication)
    workflow.add_node("retriever", retrieve_guidelines)
    workflow.add_node("generator", generate_response)
    workflow.add_node("verifier", verify_output)

    workflow.set_entry_point("topic_detector")

    workflow.add_conditional_edges("topic_detector", route_after_topic)
    workflow.add_conditional_edges("medication_gate", route_after_medication)

    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", "verifier")
    workflow.add_edge("verifier", END)

    return workflow.compile()


app: GraphApp = build_workflow()
