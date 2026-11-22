"""Unit tests for Step 9 RAG retrieval verification logic."""

import pytest

from scripts.verify_rag import RetrievedChunk, SearchResult, validate_results


def _create_dummy_result(
    sim: float,
    guideline: str = "2026 Dietary Guidance to Improve Cardiovascular Health",
    section: str = "Dietary Patterns and Fiber Intake",
    content: str = "Diets high in soluble fiber significantly reduce LDL cholesterol levels in clinical trials.",
) -> SearchResult:
    return SearchResult(
        chunk=RetrievedChunk(
            id="11111111-1111-1111-1111-111111111111",
            chunk_id="test_001",
            guideline_title=guideline,
            publication_year=2026,
            section_title=section,
            content=content,
        ),
        similarity=sim,
    )


def test_validate_results_success():
    """Verify that compliant results pass all Step 9 assertions."""
    results = [
        _create_dummy_result(0.85),
        _create_dummy_result(0.78),
        _create_dummy_result(0.72),
    ]
    # Should not raise
    validate_results(results, latency_ms=12.5, max_latency_ms=25.0)


def test_validate_results_fails_min_count():
    """Fails if fewer than 3 chunks returned."""
    results = [
        _create_dummy_result(0.85),
        _create_dummy_result(0.78),
    ]
    with pytest.raises(AssertionError, match="fewer than 3 chunks"):
        validate_results(results, latency_ms=10.0, max_latency_ms=25.0)


def test_validate_results_fails_low_similarity():
    """Fails if top similarity is below 0.70."""
    results = [
        _create_dummy_result(0.68),
        _create_dummy_result(0.65),
        _create_dummy_result(0.60),
    ]
    with pytest.raises(AssertionError, match="below 0.70 threshold"):
        validate_results(results, latency_ms=10.0, max_latency_ms=25.0)


def test_validate_results_fails_domain_mismatch():
    """Fails if top chunk is completely outside lipid/dietary clinical domain."""
    results = [
        _create_dummy_result(
            0.85,
            guideline="Software Engineering Architecture Manual",
            section="Database Sharding",
            content="This section describes database sharding strategies for high-throughput transactional systems.",
        ),
        _create_dummy_result(0.78),
        _create_dummy_result(0.72),
    ]
    with pytest.raises(AssertionError, match="does not match clinical domain"):
        validate_results(results, latency_ms=10.0, max_latency_ms=25.0)


def test_validate_results_fails_empty_content():
    """Fails if any chunk has empty or near-empty content."""
    results = [
        _create_dummy_result(0.85),
        _create_dummy_result(0.78),
        _create_dummy_result(0.72, content="too short"),
    ]
    with pytest.raises(AssertionError, match="content too short"):
        validate_results(results, latency_ms=10.0, max_latency_ms=25.0)


def test_validate_results_fails_corrupted_text():
    """Fails if any chunk contains replacement character or empty content."""
    corrupted_content = "This clinical text has replacement \xef\xbf\xbd characters and invalid formatting inside the paragraph."
    results = [
        _create_dummy_result(0.85),
        _create_dummy_result(0.78),
        _create_dummy_result(0.72, content=corrupted_content),
    ]
    with pytest.raises(AssertionError, match="contains replacement chars"):
        validate_results(results, latency_ms=10.0, max_latency_ms=25.0)


def test_validate_results_fails_latency_budget():
    """Fails if query latency exceeds budget."""
    results = [
        _create_dummy_result(0.85),
        _create_dummy_result(0.78),
        _create_dummy_result(0.72),
    ]
    with pytest.raises(AssertionError, match="exceeded budget"):
        validate_results(results, latency_ms=30.0, max_latency_ms=25.0)
