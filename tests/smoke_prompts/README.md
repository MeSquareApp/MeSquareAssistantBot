# Smoke Testing Prompts

This folder contains curated prompt suites designed for deterministic smoke testing of the `MeSquareAssistantBot` clinical reasoning graph.

## Suite Categories

| File | Category | Target Guardrail / Node | Expected Status |
| --- | --- | --- | --- |
| `emergency.json` | Acute Medical Emergency | `topic_detector` | `emergency` |
| `off_topic.json` | Non-Cardiovascular Queries | `topic_detector` | `off_topic` |
| `medication_gate.json` | Prescription Drug Requests | `medication_gate` | `medication_prohibited` |
| `lifestyle_guidance.json` | Cardiovascular Lifestyle | `retriever` -> `generator` -> `verifier` | `success` |

## Prompt Object Schema

Each test case contains:
- `id`: Unique identifier (e.g., `EMERG-001`, `MED-001`, `LIFE-001`).
- `prompt`: Inbound user query text.
- `description`: Clinical rationale and scenario description.
- `expected_status`: Expected terminal state (`emergency`, `off_topic`, `medication_prohibited`, `success`).
- `expected_keyword`: Substring required in the assistant output.
- `expect_citations`: Boolean indicating whether guideline citations must be returned.

## Execution

Run all smoke suites:
```bash
uv run python scripts/smoke_test.py
```

Run a specific category:
```bash
uv run python scripts/smoke_test.py --category emergency
uv run python scripts/smoke_test.py --category medication
uv run python scripts/smoke_test.py --category off_topic
uv run python scripts/smoke_test.py --category lifestyle
```

Test an arbitrary prompt ad-hoc:
```bash
uv run python scripts/smoke_test.py --prompt "I have severe chest pressure"
```
