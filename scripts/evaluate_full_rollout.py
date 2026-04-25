from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from construction_safety_env.env import ConstructionSafetyEnv
from construction_safety_env.models import ConstruxAction
from inference import _heuristic_action


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUT_DIR = Path("demo_artifacts")
DIFFICULTIES = ("easy", "medium", "hard")
POLICY_ORDER = ("random", "heuristic", "sft", "grpo")
COLORS = {
    "random": "#b44d68",
    "heuristic": "#237a71",
    "sft": "#4576d1",
    "grpo": "#9b59b6",
}


def build_prompt(observation) -> str:
    tasks = "\n".join(
        f"- {task.task_id}: status={task.status}, crew={task.required_crew}, blocked={task.blocked_reasons}"
        for task in observation.tasks.values()
    )
    weather = ", ".join(
        f"day {item.day} rain={item.rain_probability} wind={item.wind_mph}"
        for item in observation.weather_forecast
    )
    return (
        "Return exactly one compact JSON object and nothing else.\n"
        "No markdown. No prose. No code fences.\n"
        "Always include action_type. Only include fields needed for that action.\n"
        '{"action_type":"check_weather"}\n'
        '{"action_type":"request_permit","permit_type":"building"}\n'
        '{"action_type":"assign_crew","crew_id":"structural","task_id":"site_survey"}\n'
        '{"action_type":"hold_crew","crew_id":"structural","reason":"Waiting for permit approval."}\n'
        f"Day {observation.day}/{observation.max_days}; budget={observation.remaining_budget}\n"
        f"Weather: {weather}\n"
        f"Inventory: {observation.inventory}\n"
        f"Permits: {[f'{k}:{v.status}' for k, v in observation.permits.items()]}\n"
        f"Alerts: {[alert.model_dump() for alert in observation.osha_alerts]}\n"
        f"Tasks:\n{tasks}\n"
    )


def extract_json(text: str) -> dict:
    text = str(text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found.")
    return json.loads(text[start : end + 1])


def random_action(observation, rng: random.Random) -> ConstruxAction:
    choices: List[ConstruxAction] = [
        ConstruxAction(action_type="check_weather"),
        ConstruxAction(action_type="check_inventory"),
    ]
    permit_types = list(observation.permits.keys())
    crew_ids = list(observation.crews.keys())
    task_ids = list(observation.tasks.keys())
    zones = sorted({task.zone for task in observation.tasks.values()})
    quote_ids = list(observation.subcontractor_quotes.keys())

    if permit_types:
        choices.append(ConstruxAction(action_type="request_permit", permit_type=rng.choice(permit_types)))
    if crew_ids:
        choices.append(
            ConstruxAction(
                action_type="hold_crew",
                crew_id=rng.choice(crew_ids),
                reason="Random baseline wait.",
            )
        )
    if crew_ids and task_ids:
        choices.append(
            ConstruxAction(
                action_type="assign_crew",
                crew_id=rng.choice(crew_ids),
                task_id=rng.choice(task_ids),
            )
        )
    if zones:
        choices.append(ConstruxAction(action_type="request_inspection", zone=rng.choice(zones)))
    if quote_ids:
        choices.append(ConstruxAction(action_type="accept_quote", quote_id=rng.choice(quote_ids)))
    choices.append(ConstruxAction(action_type="order_material", material="concrete", quantity=1, quality="cheap"))
    return rng.choice(choices)


def load_adapter(adapter_path: Path):
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Missing local adapter folder: {adapter_path}. Run training and evaluation in the same Colab session, "
            "or upload the saved adapter directory before evaluating."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_kwargs = {"device_map": "auto"}
    if "dtype" in AutoModelForCausalLM.from_pretrained.__code__.co_varnames:
        model_kwargs["dtype"] = dtype
    else:
        model_kwargs["torch_dtype"] = dtype
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **model_kwargs)
    model = PeftModel.from_pretrained(base, str(adapter_path), local_files_only=True)
    model.eval()
    return model, tokenizer


def model_action(model, tokenizer, observation) -> ConstruxAction:
    prompt = build_prompt(observation)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return ConstruxAction.model_validate(extract_json(generated))


def rollout(
    policy_name: str,
    difficulty: str,
    seed: int,
    max_steps: int,
    model=None,
    tokenizer=None,
) -> Dict[str, Any]:
    env = ConstructionSafetyEnv()
    observation = env.reset(difficulty=difficulty, seed=seed)
    rng = random.Random(seed)
    rewards: List[float] = []
    invalid_actions = 0
    generation_failures = 0

    try:
        while not observation.done and env.state().step_index < max_steps:
            try:
                if policy_name == "heuristic":
                    action = _heuristic_action(observation)
                elif policy_name == "random":
                    action = random_action(observation, rng)
                else:
                    action = model_action(model, tokenizer, observation)
            except Exception:
                generation_failures += 1
                invalid_actions += 1
                action = ConstruxAction(action_type="check_weather")

            observation, reward, done, _info = env.step(action)
            rewards.append(reward.value)
            if observation.last_action_error:
                invalid_actions += 1
            if done:
                break

        state = env.state()
        return {
            "policy": policy_name,
            "difficulty": difficulty,
            "seed": seed,
            "success": state.success,
            "score": round(state.current_score, 4),
            "steps": state.step_index,
            "total_reward": round(sum(rewards), 4),
            "invalid_actions": invalid_actions,
            "generation_failures": generation_failures,
            "safety_violations": state.safety_violations,
            "missing_incident_reports": state.missing_incident_reports,
        }
    finally:
        env.close()


def summarize(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    rows = list(rows)
    summary: Dict[str, Dict[str, float]] = {}
    for policy in POLICY_ORDER:
        summary[policy] = {}
        for difficulty in DIFFICULTIES:
            values = [row["score"] for row in rows if row["policy"] == policy and row["difficulty"] == difficulty]
            summary[policy][difficulty] = round(sum(values) / len(values), 4)
    return summary


def summarize_diagnostics(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    rows = list(rows)
    diagnostics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for policy in POLICY_ORDER:
        diagnostics[policy] = {}
        for difficulty in DIFFICULTIES:
            matching = [row for row in rows if row["policy"] == policy and row["difficulty"] == difficulty]
            count = max(1, len(matching))
            diagnostics[policy][difficulty] = {
                "success_rate": round(sum(1 for row in matching if row["success"]) / count, 4),
                "avg_steps": round(sum(float(row["steps"]) for row in matching) / count, 2),
                "avg_invalid_actions": round(sum(float(row["invalid_actions"]) for row in matching) / count, 2),
                "avg_generation_failures": round(sum(float(row["generation_failures"]) for row in matching) / count, 2),
            }
    return diagnostics


def write_svg(summary: Dict[str, Dict[str, float]], path: Path, title: str, subtitle: str) -> None:
    width = 1280
    height = 720
    left = 104
    right = 74
    top = 168
    bottom = 102
    plot_height = height - top - bottom
    plot_width = width - left - right
    bar_width = 54
    inner_gap = 12
    group_width = len(POLICY_ORDER) * bar_width + (len(POLICY_ORDER) - 1) * inner_gap
    group_gap = (plot_width - len(DIFFICULTIES) * group_width) / (len(DIFFICULTIES) + 1)

    x = left + group_gap
    bars = []
    labels = []
    for difficulty in DIFFICULTIES:
        group_start = x
        for policy in POLICY_ORDER:
            value = summary[policy][difficulty]
            bar_height = int(plot_height * value)
            y = height - bottom - bar_height
            bars.append(
                f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" rx="2" fill="{COLORS[policy]}" />'
            )
            labels.append(
                f'<text x="{x + bar_width / 2}" y="{y - 10}" text-anchor="middle" font-size="14" font-weight="600">{value:.2f}</text>'
            )
            x += bar_width + inner_gap
        group_center = group_start + ((bar_width + inner_gap) * len(POLICY_ORDER) - inner_gap) / 2
        labels.append(
            f'<text x="{group_center}" y="{height - 44}" text-anchor="middle" font-size="18" font-weight="700" font-family="Arial, sans-serif">{difficulty.title()}</text>'
        )
        x += group_gap

    legend_items = []
    lx = left
    ly = 108
    for policy, label in (
        ("random", "Random"),
        ("heuristic", "Heuristic"),
        ("sft", "SFT"),
        ("grpo", "GRPO"),
    ):
        legend_items.append(f'<rect x="{lx}" y="{ly}" width="18" height="18" rx="2" fill="{COLORS[policy]}" />')
        legend_items.append(f'<text x="{lx + 28}" y="{ly + 15}" font-size="16" font-family="Arial, sans-serif">{label}</text>')
        lx += 150

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fbfaf6" />
  <text x="{left}" y="48" font-size="30" font-weight="700" font-family="Arial, sans-serif">{title}</text>
  <text x="{left}" y="78" font-size="16" font-family="Arial, sans-serif" fill="#444">{subtitle}</text>
  <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#333" stroke-width="1.5" />
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#333" stroke-width="1.5" />
  <line x1="{left}" y1="{top}" x2="{width - right}" y2="{top}" stroke="#ddd" />
  <line x1="{left}" y1="{top + plot_height / 2}" x2="{width - right}" y2="{top + plot_height / 2}" stroke="#e6e1d8" />
  <text x="58" y="{top + 5}" font-size="14" font-family="Arial, sans-serif">1.0</text>
  <text x="58" y="{top + plot_height / 2 + 5}" font-size="14" font-family="Arial, sans-serif">0.5</text>
  <text x="58" y="{height - bottom + 5}" font-size="14" font-family="Arial, sans-serif">0.0</text>
  <text x="30" y="{top + plot_height / 2}" transform="rotate(-90 30 {top + plot_height / 2})" text-anchor="middle" font-size="15" font-family="Arial, sans-serif" fill="#444">Average score</text>
  {''.join(bars)}
  {''.join(labels)}
  {''.join(legend_items)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate random, heuristic, SFT, and GRPO with full-episode control.")
    parser.add_argument("--sft-path", default="construx-rl-sft", help="Path to the saved SFT adapter folder.")
    parser.add_argument("--grpo-path", default="construx-rl-grpo", help="Path to the saved GRPO adapter folder.")
    parser.add_argument("--seed-count", type=int, default=5, help="Number of seeds per difficulty.")
    parser.add_argument("--max-steps", type=int, default=90, help="Hard cap on environment steps per episode.")
    parser.add_argument(
        "--output-prefix",
        default="full_rollout",
        help="Prefix for generated files inside demo_artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    if args.max_steps < 55:
        print(
            f"warning: --max-steps={args.max_steps} is a quick smoke-test cap. "
            "Use --max-steps 90 for the real full-rollout comparison."
        )

    sft_model, sft_tokenizer = load_adapter(Path(args.sft_path))
    grpo_model, grpo_tokenizer = load_adapter(Path(args.grpo_path))

    rows: List[Dict[str, Any]] = []
    for difficulty in DIFFICULTIES:
        for seed in range(args.seed_count):
            rows.append(rollout("random", difficulty, seed, args.max_steps))
            rows.append(rollout("heuristic", difficulty, seed, args.max_steps))
            rows.append(rollout("sft", difficulty, seed, args.max_steps, sft_model, sft_tokenizer))
            rows.append(rollout("grpo", difficulty, seed, args.max_steps, grpo_model, grpo_tokenizer))

    summary = summarize(rows)
    diagnostics = summarize_diagnostics(rows)
    csv_path = OUT_DIR / f"{args.output_prefix}_comparison.csv"
    json_path = OUT_DIR / f"{args.output_prefix}_summary.json"
    svg_path = OUT_DIR / f"{args.output_prefix}_comparison.svg"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(
        json.dumps(
            {
                "average_scores": summary,
                "diagnostics": diagnostics,
                "note": "Use diagnostics to spot failed model rollouts. High generation_failures means the model output could not be parsed into a ConstruxAction.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_svg(
        summary,
        svg_path,
        title="Construx-RL Full-Rollout Comparison",
        subtitle="Average final score when each policy controls the entire episode without heuristic rescue.",
    )

    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {svg_path}")
    print(json.dumps({"average_scores": summary, "diagnostics": diagnostics}, indent=2))


if __name__ == "__main__":
    main()
