---
title: Construx-RL
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - construction
  - multi-agent
  - long-horizon
  - enterprise
  - osha
---

# Construx-RL

Construx-RL is an OpenEnv reinforcement learning environment where an LLM acts as a construction site manager. The agent must deliver a building project across permits, material lead times, weather, OSHA safety rules, budget pressure, crew coordination, and subcontractor negotiation.

The core failure mode we target is familiar: LLMs can produce confident plans that violate physical dependencies, ignore delayed approvals, miss safety constraints, or spend the budget into failure. Construx-RL turns those mistakes into deterministic environment feedback.

## Hackathon Themes

- Theme 1, Multi-Agent Interactions: structural, MEP, finishing, admin, and subcontractor agents have different constraints, costs, availability, and rejection behavior.
- Theme 2, Long-Horizon Planning: the hard scenario runs across 30 simulated days, with permits taking 2-3 days and materials arriving 3 days after ordering.
- Theme 3.1, Professional World Modeling: the agent interacts with permits, inventory, weather, budget, inspections, OSHA incidents, and quote negotiation.
- Theme 4, Self-Improvement: episode memory and curriculum difficulty are built in; easy unlocks medium, then hard as rewards improve.

## Scenarios

- Easy: 5 tasks, one zone, one building permit, stable weather, basic OSHA and material timing.
- Medium: 10 tasks, 4 zones, weather disruption, 3 permits, supply planning, and MEP/roof coordination.
- Hard: 15 tasks, 5 zones, 30 simulated days, random weather, OSHA events, subcontractor welding, budget pressure, and final inspection.

## Environment API

The environment exposes OpenEnv-style `reset()`, `step(action)`, and `state()` methods through both Python and FastAPI.

Action types:

```text
assign_crew
hold_crew
order_material
check_inventory
check_weather
request_permit
check_permit_status
file_incident_report
request_inspection
request_quote
accept_quote
negotiate
```

Observation includes:

- current day, max days, remaining budget
- full task DAG status and blocked reasons
- crew availability and assignments
- 3-day weather forecast
- inventory and pending material order ETAs
- permit status
- inspected zones
- active OSHA alerts and incident report status
- subcontractor quotes
- compressed site log and cross-episode memory hint

## Hard Rules

- Walls/framing cannot start before foundation cure.
- Concrete cannot be poured in unsafe rain.
- Crane work is blocked above 25 mph wind.
- Roof and rough-in gate interior work.
- Materials arrive 3 days after ordering.
- No work starts without required approved permit.
- Budget reaching zero fails the project.

OSHA rules encoded:

- `OSHA 1926.502`: fall protection above 6 feet
- `OSHA 1926.652`: shoring for excavations deeper than 5 feet
- `OSHA 1926.550`: crane swing radius and unsafe crane conditions
- `OSHA 1910.147`: lockout/tagout for electrical work
- `OSHA 1926.100`: hard hats in active zones
- `OSHA 1926.451`: scaffold inspection
- `OSHA 1926.150`: fire extinguisher for welding
- `OSHA 1926.32`: valid permit requirement

## Reward Functions

Construx-RL reports independent reward components:

- Progress: `+0.1` per task completed in dependency-valid order.
- Budget efficiency: remaining budget divided by starting budget at completion.
- Safety: `-0.5` per OSHA violation, `-0.3` per missing incident report, `+0.3` for zero-incident completion.
- Schedule: `+0.5` if completed before day 28, `+0.2` if completed day 28-30.
- Anti-hack checks: penalties for invalid actions, budget failure, and timeout.

## Run Locally

```bash
pip install -r requirements.txt
python inference.py
```

Run the server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Then call:

```bash
curl http://127.0.0.1:7860/health
curl http://127.0.0.1:7860/schema
```

## Demo Trace

`inference.py` prints the required format:

```text
[START] env=construx_rl difficulty=hard model=Qwen/Qwen2.5-0.5B-Instruct
[STEP] step=1 day=1 action=request_permit(permit_type='building') reward=0.000 done=false error=null
[END] success=true steps=55 score=0.755 rewards=...
```

The included heuristic baseline is intentionally simple but completes all three scenarios, giving you a reproducible demo and a sanity check for reward curves.

## Training

The Colab-oriented training skeleton is in:

```text
scripts/train_grpo_colab.py
```

It uses:

- OpenEnv-compatible environment loop
- TRL `GRPOTrainer`
- Unsloth 4-bit loading and LoRA
- Qwen2.5-0.5B-Instruct by default for Colab-friendly GRPO iteration
- verifier-style reward from the actual Construx-RL environment

For the hackathon Colab, install the package from the Hugging Face Space repo, then run the script cells after adding your HF/W&B credentials. If you want to evaluate against a larger hosted model, override `MODEL_NAME` in `inference.py` via environment variable.

## Deployment

This repo is ready for Hugging Face Spaces with Docker:

- `Dockerfile`
- `openenv.yaml`
- `server/app.py`

The Space exposes `/reset`, `/step`, `/state`, `/schema`, `/tasks`, `/health`, and `/healthz`.
