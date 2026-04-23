from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.app import app
from inference import run_episode


def check_api() -> None:
    client = TestClient(app)
    reset = client.post("/reset", json={"difficulty": "easy", "seed": 0})
    reset.raise_for_status()
    body = reset.json()
    session_id = body["session_id"]
    observation = body["observation"]
    assert observation["benchmark_name"] == "construx_rl"
    assert observation["difficulty"] == "easy"

    step = client.post(
        "/step",
        json={
            "session_id": session_id,
            "action": {"action_type": "request_permit", "permit_type": "building"},
        },
    )
    step.raise_for_status()
    step_body = step.json()
    assert "reward" in step_body
    assert "components" in step_body["reward"]
    assert step_body["done"] is False
    print("api_reset_step_reward: PASS")


def check_scenarios() -> None:
    for difficulty in ("easy", "medium", "hard"):
        result = run_episode(difficulty, None)
        assert result["success"] is True, f"{difficulty} did not complete"
        assert result["score"] > 0.70, f"{difficulty} score too low: {result['score']}"
        print(f"scenario_{difficulty}: PASS score={result['score']:.3f} steps={result['steps']}")


def main() -> None:
    check_api()
    check_scenarios()
    print("demo_check: PASS")


if __name__ == "__main__":
    main()
