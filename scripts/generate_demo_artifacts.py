from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from construction_safety_env.env import ConstructionSafetyEnv
from construction_safety_env.models import ConstruxAction
from inference import _heuristic_action


OUT_DIR = Path("demo_artifacts")
DIFFICULTIES = ("easy", "medium", "hard")
RANDOM_ACTION_TYPES = (
    "check_weather",
    "check_inventory",
    "request_permit",
    "assign_crew",
    "order_material",
    "request_inspection",
    "hold_crew",
)


def random_action(observation, rng: random.Random) -> ConstruxAction:
    action_type = rng.choice(RANDOM_ACTION_TYPES)
    if action_type == "request_permit":
        permit_type = rng.choice(list(observation.permits) or ["building"])
        return ConstruxAction(action_type="request_permit", permit_type=permit_type)
    if action_type == "assign_crew":
        crew = rng.choice(list(observation.crews.values()))
        task = rng.choice(list(observation.tasks.values()))
        return ConstruxAction(action_type="assign_crew", crew_id=crew.crew_id, task_id=task.task_id)
    if action_type == "order_material":
        return ConstruxAction(action_type="order_material", material="concrete", quantity=1, quality="cheap")
    if action_type == "request_inspection":
        task = rng.choice(list(observation.tasks.values()))
        return ConstruxAction(action_type="request_inspection", zone=task.zone)
    if action_type == "hold_crew":
        crew = rng.choice(list(observation.crews.values()))
        return ConstruxAction(action_type="hold_crew", crew_id=crew.crew_id, reason="Random baseline wait.")
    return ConstruxAction(action_type=action_type)


def run_policy(
    difficulty: str,
    seed: int,
    policy: Callable,
    max_steps: int = 90,
) -> Dict[str, object]:
    env = ConstructionSafetyEnv()
    observation = env.reset(difficulty=difficulty, seed=seed)
    rewards: List[float] = []
    rng = random.Random(seed)
    for _ in range(max_steps):
        action = policy(observation, rng)
        observation, reward, done, _info = env.step(action)
        rewards.append(reward.value)
        if done:
            break
    state = env.state()
    return {
        "difficulty": difficulty,
        "seed": seed,
        "success": state.success,
        "score": state.current_score,
        "steps": state.step_index,
        "total_reward": round(sum(rewards), 4),
        "safety_violations": state.safety_violations,
        "missing_incident_reports": state.missing_incident_reports,
    }


def heuristic_policy(observation, _rng: random.Random) -> ConstruxAction:
    return _heuristic_action(observation)


def write_csv(rows: List[Dict[str, object]]) -> None:
    path = OUT_DIR / "policy_comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(summary: Dict[str, object]) -> None:
    path = OUT_DIR / "policy_comparison_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_svg(summary: Dict[str, Dict[str, float]]) -> None:
    width = 960
    height = 560
    left = 92
    right = 52
    top = 118
    bottom = 82
    plot_height = height - top - bottom
    bar_width = 64
    gap = 56
    max_score = 1.0
    labels = list(DIFFICULTIES)
    colors = {"random_baseline": "#b44d68", "heuristic_demo": "#237a71"}
    x = left + 54
    bars = []
    text = []
    for difficulty in labels:
        group_start = x
        for policy_name in ("random_baseline", "heuristic_demo"):
            value = summary[policy_name][difficulty]
            bar_height = int(plot_height * value / max_score)
            y = height - bottom - bar_height
            bars.append(
                f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" rx="2" fill="{colors[policy_name]}" />'
            )
            text.append(f'<text x="{x + bar_width / 2}" y="{y - 12}" text-anchor="middle" font-size="15" font-weight="600">{value:.2f}</text>')
            x += bar_width + 12
        group_center = group_start + bar_width + 6
        text.append(f'<text x="{group_center}" y="{height - 34}" text-anchor="middle" font-size="17" font-weight="600">{difficulty.title()}</text>')
        x += gap

    legend = (
        '<rect x="690" y="38" width="18" height="18" rx="2" fill="#b44d68" />'
        '<text x="718" y="53" font-size="16">Random baseline</text>'
        '<rect x="690" y="68" width="18" height="18" rx="2" fill="#237a71" />'
        '<text x="718" y="83" font-size="16">Demo policy</text>'
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fbfaf6" />
  <text x="{left}" y="48" font-size="28" font-weight="700" font-family="Arial, sans-serif">Construx-RL Reward Comparison</text>
  <text x="{left}" y="78" font-size="15" font-family="Arial, sans-serif" fill="#444">Evaluation artifact: random actions vs deterministic demo policy. Replace demo policy with trained model after GRPO.</text>
  <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#333" stroke-width="1.5" />
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#333" stroke-width="1.5" />
  <line x1="{left}" y1="{top}" x2="{width - right}" y2="{top}" stroke="#ddd" />
  <line x1="{left}" y1="{top + plot_height / 2}" x2="{width - right}" y2="{top + plot_height / 2}" stroke="#e6e1d8" />
  <text x="48" y="{top + 5}" font-size="14" font-family="Arial, sans-serif">1.0</text>
  <text x="48" y="{top + plot_height / 2 + 5}" font-size="14" font-family="Arial, sans-serif">0.5</text>
  <text x="48" y="{height - bottom + 5}" font-size="14" font-family="Arial, sans-serif">0.0</text>
  <text x="22" y="{top + plot_height / 2}" transform="rotate(-90 22 {top + plot_height / 2})" text-anchor="middle" font-size="14" font-family="Arial, sans-serif" fill="#444">Average score</text>
  {''.join(bars)}
  {''.join(text)}
  {legend}
</svg>
"""
    (OUT_DIR / "reward_comparison.svg").write_text(svg, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    rows: List[Dict[str, object]] = []
    for policy_name, policy in (("random_baseline", random_action), ("heuristic_demo", heuristic_policy)):
        for difficulty in DIFFICULTIES:
            for seed in range(5):
                row = run_policy(difficulty, seed, policy)
                row["policy"] = policy_name
                rows.append(row)

    averages: Dict[str, Dict[str, float]] = {"random_baseline": {}, "heuristic_demo": {}}
    for policy_name in averages:
        for difficulty in DIFFICULTIES:
            scores = [float(row["score"]) for row in rows if row["policy"] == policy_name and row["difficulty"] == difficulty]
            averages[policy_name][difficulty] = round(sum(scores) / len(scores), 4)

    summary = {
        "important_note": "This is a demo/evaluation artifact, not proof of completed RL training. Use scripts/train_grpo_colab.py for actual GRPO training, then replace heuristic_demo with trained_model.",
        "average_scores": averages,
        "runs": rows,
    }
    write_csv(rows)
    write_json(summary)
    write_svg(averages)
    print(f"wrote {OUT_DIR / 'policy_comparison.csv'}")
    print(f"wrote {OUT_DIR / 'policy_comparison_summary.json'}")
    print(f"wrote {OUT_DIR / 'reward_comparison.svg'}")
    print(json.dumps(summary["average_scores"], indent=2))


if __name__ == "__main__":
    main()
