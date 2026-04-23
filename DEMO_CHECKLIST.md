# Construx-RL Judge Checklist

## 1. Is it actually runnable?

Yes. Run the API server:

```powershell
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
$env:API_BASE_URL="http://127.0.0.1:8000"
python inference.py
```

Expected result: easy, medium, and hard episodes end with `success=true`.

## 2. Can judges see reset, step, and reward live?

Reset:

```powershell
$reset = Invoke-RestMethod -Method Post http://127.0.0.1:8000/reset -ContentType "application/json" -Body '{"difficulty":"easy","seed":0}'
$reset.observation.benchmark_name
```

Step:

```powershell
$sid = $reset.session_id
Invoke-RestMethod -Method Post http://127.0.0.1:8000/step -ContentType "application/json" -Body "{`"session_id`":`"$sid`",`"action`":{`"action_type`":`"request_permit`",`"permit_type`":`"building`"}}"
```

The response includes:

- `observation`
- `reward.value`
- `reward.components`
- `done`
- `info`

## 3. Did training improve performance?

Actual GRPO training is prepared but not yet run in this repo.

Training skeleton:

```text
scripts/train_grpo_colab.py
```

Demo/evaluation graph generator:

```powershell
python scripts/generate_demo_artifacts.py
```

Generated files:

```text
demo_artifacts/policy_comparison.csv
demo_artifacts/policy_comparison_summary.json
demo_artifacts/reward_comparison.svg
```

Important honest wording:

> The current graph compares a random baseline against the deterministic demo policy. It proves the environment has a measurable reward signal and a solvable task. Actual RL improvement should be shown after running `scripts/train_grpo_colab.py` on hackathon compute and replacing the demo policy with the trained model.

## 4. Is it OpenEnv compliant?

Files/endpoints present:

- `openenv.yaml`
- `Dockerfile`
- `/reset`
- `/step`
- `/state`
- `/schema`
- `/tasks`
- `/health`

Validation command to try:

```powershell
openenv validate .
```

If unavailable:

```powershell
python -m openenv validate .
```

If that says `openenv` has no `__main__`, use the actual console entrypoint module:

```powershell
python -m openenv.cli.__main__ validate .
```

## 5. How much is real vs mocked?

Honest answer:

> Construx-RL is a deterministic simulation environment. Weather, permit approvals, material delays, crew costs, OSHA violations, and subcontractor quotes are simulated rather than connected to external production APIs. This is intentional for RL training because verifiable deterministic rewards are easier to optimize, reproduce, and audit. The FastAPI/OpenEnv interaction is real, and every action changes environment state.

## 6. End-to-end MVP test

```powershell
python scripts/demo_check.py
```

Expected final line:

```text
demo_check: PASS
```
