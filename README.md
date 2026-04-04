---
title: Construction Site Safety Inspector
sdk: docker
app_port: 7860
tags:
  - openenv
  - construction
  - safety
---

# Construction Site Safety Inspector

`construction_site_safety_inspector` is an OpenEnv-style benchmark for a real job humans actually do: reading construction site reports, identifying safety violations, and issuing the right OSHA citations with practical abatement guidance.

The environment is designed for agent training and evaluation in safety-heavy operational work. Instead of solving a toy task, the agent has to interpret messy natural-language reports, separate overlapping hazards, pick the correct OSHA rule, and avoid inventing unsupported citations.

## Why this environment is useful

Construction safety inspection is a strong real-world agent domain because:

- Site observations arrive as natural-language narratives, not clean labels.
- The agent must map evidence to formal standards.
- Partial progress matters. Finding one real hazard is valuable even if the report is incomplete.
- Wrong or hallucinated citations should be penalized, not rewarded.

This environment models those constraints with deterministic graders backed by official OSHA construction standards.

## Task set

There are three graded tasks with increasing difficulty:

1. `easy_roof_fall_protection`
   Residential roofing report with two obvious hazards: fall protection at an 18-foot roof edge and improper ladder extension.
2. `medium_trench_excavation_control`
   Utility trench report with three interacting excavation hazards: missing cave-in protection, inadequate egress, and spoil piles at the edge.
3. `hard_scaffold_multi_hazard`
   Mixed scaffold and interior work report with multiple simultaneous fall hazards: missing scaffold fall protection, falling-object exposure, and an uncovered floor opening.

## OSHA rules encoded in the grader

The tasks use official OSHA construction rules as the grading backbone:

- `29 CFR 1926.501(b)(1)` unprotected sides and edges
- `29 CFR 1926.501(b)(4)(i)` floor holes
- `29 CFR 1926.1053(b)(1)` ladder side rails at upper landing
- `29 CFR 1926.651(c)(2)` trench egress within 25 feet
- `29 CFR 1926.651(j)(2)` spoil piles at least 2 feet from excavation edge
- `29 CFR 1926.652(a)(1)` excavation cave-in protection
- `29 CFR 1926.451(g)(1)` scaffold fall protection above 10 feet
- `29 CFR 1926.451(h)(1)` scaffold falling object protection

Official sources:

- https://www.osha.gov/laws-regs/regulations/standardnumber/1926/1926.501
- https://www.osha.gov/laws-regs/regulations/standardnumber/1926/1926.1053
- https://www.osha.gov/laws-regs/regulations/standardnumber/1926/1926.651
- https://www.osha.gov/laws-regs/regulations/standardnumber/1926/1926.652
- https://www.osha.gov/laws-regs/regulations/standardnumber/1926/1926.451

## API design

The environment exposes standard `reset()`, `step()`, and `state()` methods through the Python environment class and HTTP endpoints.

### Action space

Typed Pydantic action model: `ConstructionSafetyAction`

- `action_type="issue_finding"`
  Submit one hazard finding with:
  `hazard_label`, `osha_citation`, `severity`, `evidence`, `corrective_action`, `confidence`
- `action_type="submit_report"`
  End the episode and trigger final grading

### Observation space

Typed Pydantic observation model: `ConstructionSafetyObservation`

Each observation includes:

- task metadata and difficulty
- inspector role and objective
- full site report text
- OSHA reference library available to the agent
- submitted findings so far
- feedback history from the deterministic grader
- current score, best score, step count, and final/done flag
- `last_action_error` for malformed or stale actions

### State space

Typed Pydantic state model: `ConstructionSafetyState`

The full internal state includes:

- current submitted findings and feedback history
- hidden target finding IDs
- current and best deterministic score
- step counters and done status

## Reward shaping

The reward is meaningful across the whole trajectory, not just at the end.

- Each `issue_finding` action is scored against the hidden target hazards.
- The step reward is the positive score gain from that action.
- Correct partial findings receive partial credit.
- Duplicate, hallucinated, and low-quality findings fail to increase reward.
- Final score includes penalties for excess steps, unmatched submissions, and duplicate findings.

This gives the agent dense signal while still preserving a deterministic final grader in the `[0.0, 1.0]` range.

## Project layout

```text
.
├── construction_safety_env/
│   ├── client.py
│   ├── env.py
│   ├── grader.py
│   ├── models.py
│   └── tasks.py
├── server/
│   └── app.py
├── Dockerfile
├── inference.py
├── openenv.yaml
├── README.md
├── requirements.txt
└── scripts/
    └── validate-submission.sh
```

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the API server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Smoke test:

```bash
curl http://127.0.0.1:7860/health
curl http://127.0.0.1:7860/healthz
curl http://127.0.0.1:7860/tasks
```

## Docker

Build and run locally:

```bash
docker build -t construction-safety-env .
docker run --rm -p 7860:7860 construction-safety-env
```

The container exposes the FastAPI app on port `7860`, which matches Hugging Face Spaces container expectations.

## Hugging Face Spaces deployment

This repository is ready for a Docker Space:

1. Create a new Hugging Face Space with SDK `Docker`.
2. Push this repository.
3. Add the `openenv` tag in the Space metadata.
4. Set any optional environment variables for baseline inference: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`.

The root endpoint returns `200`, and `POST /reset` creates a live environment session suitable for validator pings.

## Baseline inference

The required inference script is at the repository root.

Environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

The script:

- uses the OpenAI client for model calls when credentials are present
- falls back to a deterministic heuristic inspector for local smoke testing
- emits strict `[START]`, `[STEP]`, and `[END]` logs per task
- runs all three tasks in sequence

Example:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="..."
python inference.py
```

## Expected baseline behavior

With the built-in heuristic fallback, the environment is designed to produce near-perfect deterministic scores on all three tasks because the scripted policy captures the benchmark's intended solution path.

Expected heuristic scores:

- `easy_roof_fall_protection`: `0.93`
- `medium_trench_excavation_control`: `0.93`
- `hard_scaffold_multi_hazard`: `0.93`
- average: `0.93`

LLM scores depend on the selected remote model, but the prompt and reward shaping are deterministic.

## Validation checklist

- `openenv.yaml` included at repo root
- typed action, observation, reward, and state models implemented with Pydantic
- `step()`, `reset()`, and `state()` implemented in `construction_safety_env/env.py`
- root `inference.py` uses the OpenAI client and required environment variables
- containerized `Dockerfile` for local and HF Spaces deployment
- optional local validator included at `scripts/validate-submission.sh`
