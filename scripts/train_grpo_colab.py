"""
Colab-friendly GRPO training for Construx-RL.

Fixes versus the original hackathon stub:
- Qwen2.5-1.5B-Instruct by default instead of 0.5B
- 120 GRPO steps instead of 20
- reward shaped by actual environment score projection, not JSON-only checks
- expects an SFT adapter folder from scripts/train_sft_colab.py

Typical flow:
    !PYTHONPATH=. python scripts/train_sft_colab.py
    !PYTHONPATH=. python scripts/train_grpo_colab.py
"""

from __future__ import annotations

import inspect
import json
import os
import re
from pathlib import Path
from typing import Any, List, Tuple

from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from construction_safety_env.env import ConstructionSafetyEnv
from construction_safety_env.models import ConstruxAction
from inference import _heuristic_action


MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
SFT_ADAPTER_PATH = os.getenv("SFT_ADAPTER_PATH", "construx-rl-sft")
OUTPUT_DIR = os.getenv("GRPO_OUTPUT_DIR", "construx-rl-grpo")
MAX_STEPS = int(os.getenv("GRPO_MAX_STEPS", "120"))
SEEDS_PER_DIFFICULTY = int(os.getenv("GRPO_SEEDS_PER_DIFFICULTY", "8"))
ROLLOUT_LIMIT = int(os.getenv("GRPO_ROLLOUT_LIMIT", "16"))


def observation_prompt(observation) -> str:
    tasks = "\n".join(
        f"{task.task_id}: {task.status}; blocked={task.blocked_reasons}; crew={task.required_crew}"
        for task in observation.tasks.values()
    )
    permits = ", ".join(f"{name}:{permit.status}" for name, permit in observation.permits.items())
    weather = ", ".join(
        f"day {item.day} rain={item.rain_probability} wind={item.wind_mph}" for item in observation.weather_forecast
    )
    return (
        f"Scenario difficulty: {observation.difficulty}\n"
        f"Scenario seed: {observation.step_index}\n"
        "Return exactly one compact JSON object and nothing else.\n"
        "No markdown. No prose. No code fences.\n"
        "Always include action_type. Only include fields needed for that action.\n"
        "Optimize dependency order, permits, material lead times, OSHA safety, budget, and schedule.\n\n"
        f"Day {observation.day}/{observation.max_days}; budget={observation.remaining_budget}; permits={permits}\n"
        f"Weather: {weather}\nInventory: {observation.inventory}\nTasks:\n{tasks}\n"
        f"Alerts: {[alert.model_dump() for alert in observation.osha_alerts]}\n"
    )


def normalize_completion(text: Any) -> str:
    if isinstance(text, list):
        parts = []
        for item in text:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        text = "\n".join(parts)
    return str(text).strip()


def extract_json(text: Any) -> dict:
    text = normalize_completion(text)
    text = text.replace("```json", "```").replace("```JSON", "```")
    match = re.search(r"\{.*?\}", text, flags=re.S)
    if not match:
        raise ValueError("No JSON object found.")
    return json.loads(match.group(0))


def parse_prompt_metadata(prompt: str) -> Tuple[str, int]:
    difficulty = "easy"
    seed = 0
    for line in prompt.splitlines():
        if line.startswith("Scenario difficulty:"):
            difficulty = line.split(":", 1)[1].strip()
        elif line.startswith("Scenario seed:"):
            try:
                seed = int(line.split(":", 1)[1].strip())
            except ValueError:
                seed = 0
    return difficulty, seed


def rollout_projection(difficulty: str, seed: int, action: ConstruxAction) -> Tuple[float, float, int]:
    env = ConstructionSafetyEnv()
    observation = env.reset(difficulty=difficulty, seed=seed)
    total_reward = 0.0
    try:
        observation, reward, done, _info = env.step(action)
        total_reward += reward.value

        steps = 1
        while not done and steps < ROLLOUT_LIMIT:
            next_action = _heuristic_action(observation)
            observation, reward, done, _info = env.step(next_action)
            total_reward += reward.value
            steps += 1

        state = env.state()
        return total_reward, state.current_score, steps
    finally:
        env.close()


def construx_reward(completions: List[Any], prompts: List[str], **_: object) -> List[float]:
    rewards: List[float] = []
    for completion, prompt in zip(completions, prompts):
        raw = normalize_completion(completion)
        score = -0.20
        if len(raw) <= 220:
            score += 0.05
        if raw.startswith("{") and raw.endswith("}"):
            score += 0.05
        if "```" not in raw:
            score += 0.05

        try:
            difficulty, seed = parse_prompt_metadata(prompt)
            payload = extract_json(raw)
            action = ConstruxAction.model_validate(payload)
            score += 0.10

            projected_reward, projected_score, projected_steps = rollout_projection(difficulty, seed, action)
            score += 0.30 * max(-1.0, min(1.0, projected_reward))
            score += 0.70 * max(0.0, min(1.0, projected_score))
            if projected_steps >= ROLLOUT_LIMIT and projected_score < 0.2:
                score -= 0.10
        except Exception:
            pass

        rewards.append(float(max(-0.25, min(1.0, score))))
    return rewards


def build_dataset() -> Dataset:
    env = ConstructionSafetyEnv()
    prompts = []
    for difficulty in ("easy", "medium", "hard"):
        for seed in range(SEEDS_PER_DIFFICULTY):
            observation = env.reset(difficulty=difficulty, seed=seed)
            prompt = observation_prompt(observation).replace(
                f"Scenario seed: {observation.step_index}",
                f"Scenario seed: {seed}",
                1,
            )
            prompts.append({"prompt": prompt})
    env.close()
    return Dataset.from_list(prompts * 8)


def build_grpo_args(use_bf16: bool, use_fp16: bool) -> GRPOConfig:
    params = inspect.signature(GRPOConfig).parameters
    kwargs = {"output_dir": OUTPUT_DIR}

    base = {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "num_generations": 2,
        "learning_rate": 5e-6,
        "logging_steps": 5,
        "max_steps": MAX_STEPS,
        "report_to": [],
        "remove_unused_columns": False,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "temperature": 0.3,
    }
    for key, value in base.items():
        if key in params:
            kwargs[key] = value
    if "max_prompt_length" in params:
        kwargs["max_prompt_length"] = 1536
    if "max_completion_length" in params:
        kwargs["max_completion_length"] = 192
    elif "max_length" in params:
        kwargs["max_length"] = 1728
    return GRPOConfig(**kwargs)


def main() -> None:
    if not Path(SFT_ADAPTER_PATH).exists():
        raise FileNotFoundError(
            f"Missing SFT adapter folder: {SFT_ADAPTER_PATH}. "
            "Run scripts/train_sft_colab.py first in the same runtime."
        )

    tokenizer = AutoTokenizer.from_pretrained(SFT_ADAPTER_PATH, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH, is_trainable=True)
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.print_trainable_parameters()

    dataset = build_dataset()
    print(f"GRPO prompt rows: {len(dataset)}")
    args = build_grpo_args(use_bf16=use_bf16, use_fp16=use_fp16)
    trainer_kwargs = {
        "model": model,
        "args": args,
        "reward_funcs": [construx_reward],
        "train_dataset": dataset,
    }
    trainer_params = inspect.signature(GRPOTrainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = GRPOTrainer(**trainer_kwargs)
    trainer.train()
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("GRPO_DONE")


if __name__ == "__main__":
    main()
