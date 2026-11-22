"""Smoke testing CLI runner for MeSquareAssistantBot.

Loads curated clinical prompt suites from tests/smoke_prompts/, executes them
through the compiled LangGraph pipeline, and validates clinical guardrails.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import BaseModel, Field

from app.chat import app
from app.chat.schemas import (
    CitationItem,
    DigitalTwinContext,
    GraphState,
    LabAnchors,
    RiskProfile,
    WearablesSummary,
)

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "tests" / "smoke_prompts"

CATEGORY_FILE_MAP: dict[str, str] = {
    "emergency": "emergency.json",
    "off_topic": "off_topic.json",
    "medication": "medication_gate.json",
    "lifestyle": "lifestyle_guidance.json",
}


class SmokeCase(BaseModel):
    """Schema for a single smoke test prompt scenario."""

    id: str
    prompt: str
    description: str
    expected_status: str
    expected_keyword: str
    expect_citations: bool = Field(default=False)


def get_default_twin() -> DigitalTwinContext:
    """Generate default digital twin context for smoke verification."""
    return DigitalTwinContext(
        user_id="smoke_patient_001",
        age=52,
        sex="male",
        smoking_status="non-smoker",
        lab_anchors=LabAnchors(
            total_cholesterol=245.0,
            ldl=168.0,
            hdl=42.0,
            systolic_bp=138.0,
            diastolic_bp=88.0,
            fasting_glucose=98.0,
            hba1c=5.6,
        ),
        wearables_summary=WearablesSummary(
            resting_heart_rate_avg_14d=74.0,
            step_count_avg_14d=4800.0,
            sleep_hours_avg_14d=6.2,
        ),
        risk_profile=RiskProfile(
            aha_prevent_10yr_score=8.4,
            nhanes_percentile=65.0,
            risk_category="borderline",
        ),
    )


def load_cases(category: str | None = None) -> list[SmokeCase]:
    """Load smoke test cases from JSON files in tests/smoke_prompts/."""
    cases: list[SmokeCase] = []

    if category and category != "all":
        filename = CATEGORY_FILE_MAP.get(category)
        if not filename:
            raise ValueError(
                f"Unknown category '{category}'. Available: {list(CATEGORY_FILE_MAP.keys())}"
            )
        files = [PROMPTS_DIR / filename]
    else:
        files = sorted(PROMPTS_DIR.glob("*.json"))

    for filepath in files:
        if not filepath.exists():
            continue
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                cases.append(SmokeCase(**item))

    return cases


def run_single_prompt(prompt: str, twin: DigitalTwinContext) -> None:
    """Run an ad-hoc prompt and print the full pipeline execution result."""
    print(f"\n[Ad-hoc Run] Query: '{prompt}'")
    start_time = time.perf_counter()
    inputs: GraphState = {
        "query": prompt,
        "twin_context": twin,
        "history": [],
    }
    result = app.invoke(inputs)
    latency_ms = (time.perf_counter() - start_time) * 1000

    print(f"Status   : {result.get('status')}")
    print(f"Latency  : {latency_ms:.1f} ms")
    citations: list[CitationItem] = result.get("citations", [])
    print(f"Citations: {len(citations)}")
    for i, c in enumerate(citations, start=1):
        print(f"  [{i}] {c.guideline_title} ({c.section_title})")
    print("\n--- Output ---")
    print(result.get("output_text", ""))
    print("--------------\n")


def run_smoke_suite(cases: list[SmokeCase], verbose: bool = False) -> bool:
    """Execute smoke test cases and print summary scorecard."""
    twin = get_default_twin()
    passed = 0
    failed = 0

    print(f"\n{'=' * 80}")
    print(f"MeSquareAssistantBot Smoke Test Suite: {len(cases)} cases")
    print(f"{'=' * 80}\n")

    for case in cases:
        start_time = time.perf_counter()
        inputs: GraphState = {
            "query": case.prompt,
            "twin_context": twin,
            "history": [],
        }

        try:
            result = app.invoke(inputs)
            latency_ms = (time.perf_counter() - start_time) * 1000
            status = result.get("status", "")
            output_text = result.get("output_text", "")
            citations: list[CitationItem] = result.get("citations", [])

            status_ok = status == case.expected_status
            keyword_ok = case.expected_keyword.lower() in output_text.lower()
            citations_ok = (
                (len(citations) > 0) if case.expect_citations else (len(citations) == 0)
            )

            case_passed = status_ok and keyword_ok and citations_ok
            badge = "[PASS]" if case_passed else "[FAIL]"

            print(
                f"{badge} {case.id:<10} | status: {status:<22} | "
                f"citations: {len(citations)} | {latency_ms:6.1f}ms | {case.description[:45]}"
            )

            if not case_passed or verbose:
                if not status_ok:
                    print(
                        f"       Expected status: '{case.expected_status}', got '{status}'"
                    )
                if not keyword_ok:
                    print(
                        f"       Missing keyword: '{case.expected_keyword}' in output text"
                    )
                if not citations_ok:
                    print(
                        f"       Citations mismatch: expected >0={case.expect_citations}, got {len(citations)}"
                    )
                if verbose or not case_passed:
                    preview = output_text.replace("\n", " ")[:140]
                    print(f"       Preview: {preview}...")

            if case_passed:
                passed += 1
            else:
                failed += 1

        except (RuntimeError, ValueError, KeyError, OSError) as exc:
            failed += 1
            print(f"[FAIL] {case.id:<10} | EXCEPTION: {exc}")

    print(f"\n{'-' * 80}")
    print(f"Summary: {passed} passed, {failed} failed out of {len(cases)} total.")
    print(f"{'-' * 80}\n")

    return failed == 0


def main() -> None:
    """Entry point for smoke test CLI execution."""
    parser = argparse.ArgumentParser(
        description="Run clinical prompt smoke test suite against LangGraph pipeline."
    )
    parser.add_argument(
        "--category",
        choices=["emergency", "off_topic", "medication", "lifestyle", "all"],
        default="all",
        help="Filter smoke suite by category",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Execute single prompt ad-hoc instead of full suite",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed previews and diagnostic messages",
    )
    args = parser.parse_args()

    if args.prompt:
        run_single_prompt(args.prompt, get_default_twin())
        return

    cases = load_cases(category=args.category)
    success = run_smoke_suite(cases, verbose=args.verbose)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
