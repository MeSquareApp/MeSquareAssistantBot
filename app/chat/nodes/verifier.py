"""Node 5: Citation mapper and fail-safe clinical safety verifier.

Performs secondary gate checks against output text to guarantee no pharmaceutical
prescriptions were hallucinated by the generator, and structures citation metadata.
"""

from app.chat.nodes.medication_gate import DRUG_PATTERN, MEDICATION_DISCLAIMER
from app.chat.schemas import CitationItem, GraphState, RetrievedChunk


def verify_output(state: GraphState) -> dict[str, object]:
    """Verify generated response safety and map retrieved chunks to formal citations."""
    text = state.get("output_text", "")
    chunks: list[RetrievedChunk] = state.get("retrieved_chunks", [])

    # Fail-safe check: Ensure LLM generated no drug recommendations
    if DRUG_PATTERN.search(text.lower()):
        return {
            "status": "medication_prohibited",
            "output_text": MEDICATION_DISCLAIMER,
            "citations": [],
        }

    citations: list[CitationItem] = []
    for c in chunks:
        citations.append(
            CitationItem(
                chunk_id=c.chunk_id,
                guideline_title=c.guideline_title,
                section_title=c.section_title,
                excerpt=f"{c.content[:220]}...",
            )
        )

    return {
        "status": "success",
        "citations": citations,
    }
