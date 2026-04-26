from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any, List, Tuple

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from construction_safety_env.action_choices import build_action_choices, format_action_choices, parse_choice_response
from construction_safety_env.env import ConstructionSafetyEnv
from construction_safety_env.models import ConstruxAction
from inference import _heuristic_action


MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
SFT_CHOICE_ADAPTER_PATH = os.getenv("SFT_CHOICE_ADAPTER_PATH", "construx-rl-sft-choice")
OUTPUT_DIR = os.getenv("GRPO_CHOICE_OUTPUT_DIR", "construx-rl-grpo-choice")
MAX_STEPS = int(os.getenv("GRPO_CHOICE_MAX_STEPS", "150"))
SEEDS_PER_DIFFICULTY = int(os.getenv("GRPO_CHOICE_SEEDS_PER_DIFFICULTY", "10"))
ROLLOUT_LIMIT = int(os.getenv("GRPO_CHOICE_ROLLOUT_LIMIT", "16"))
MAX_COMPLETION_LENGTH = int(os.getenv("GRPO_CHOICE_MAX_COMPLETION_LENGTH", "8"))
DIFFICULTIES = tuple(item.strip() for item in os.getenv("GRPO_CHOICE_DIFFICULTIES", "easy,medium,hard").split(",") if item.strip())


def choice_prompt(observation) -> str:
    tasks = "\n".join(
        f"- {task.task_id}: status={task.status}, crew={task.required_crew}, blocked={task.blocked_reasons}"
        for task in observation.tasks.values()
    )
    weather = ", ".join(
        f"day {item.day} rain={item.rain_probability} wind={item.wind_mph}"
        for item in observation.weather_forecast
    )
    choices = build_action_choices(observation)
    return (
        f"Scenario difficulty: {observation.difficulty}\n"
        f"Scenario seed: {observation.step_index}\n"
        "You are the Construx-RL site manager.\n"
        "Choose exactly one numbered action from the candidate list.\n"
        "Return ONLY the number.\n\n"
        f"Day {observation.day}/{observation.max_days}; budget={observation.remaining_budget}\n"
        f"Weather: {weather}\n"
        f"Permits: {[f'{k}:{v.status}' for k, v in observation.permits.items()]}\n"
        f"Inventory: {observation.inventory}\n"
        f"Tasks:\n{tasks}\n\n"
        f"Candidate actions:\n{format_action_choices(choices)}\n"
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


def first_choice_action(prompt: str, completion: Any) -> ConstruxAction:
    difficulty, seed = parse_prompt_metadata(prompt)
    env = ConstructionSafetyEnv()
    observation = env.reset(difficulty=difficulty, seed=seed)
    try:
        choices = build_action_choices(observation)
        return parse_choice_response(normalize_completion(completion), choices)
    finally:
        env.close()


def rollout_projection(difficulty: str, seed: int, action: ConstruxAction) -> Tuple[float, float, int, bool]:
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
        return total_reward, state.current_score, steps, state.success
    finally:
        env.close()


def construx_choice_reward(completions: List[Any], prompts: List[str], **_: object) -> List[float]:
    rewards: List[float] = []
    for completion, prompt in zip(completions, prompts):
        raw = normalize_completion(completion)
        digits = "".join(ch for ch in raw if ch.isdigit())
        score = -0.10
        if digits:
            score += 0.10
        if raw == digits and len(raw) <= 2:
            score += 0.10
        elif len(raw) <= 8:
            score += 0.03

        try:
            difficulty, seed = parse_prompt_metadata(prompt)
            action = first_choice_action(prompt, raw)
            score += 0.15
            projected_reward, projected_score, projected_steps, success = rollout_projection(difficulty, seed, action)
            score += 0.30 * max(-1.0, min(1.0, projected_reward))
            score += 0.70 * max(0.0, min(1.0, projected_score))
            if success:
                score += 0.10
            if projected_steps >= ROLLOUT_LIMIT and projected_score < 0.2:
                score -= 0.05
        except Exception:
            pass

        rewards.append(float(max(-0.20, min(1.0, score))))
    return rewards


def build_dataset() -> Dataset:
    env = ConstructionSafetyEnv()
    prompts = []
    for difficulty in DIFFICULTIES:
        for seed in range(SEEDS_PER_DIFFICULTY):
            observation = env.reset(difficulty=difficulty, seed=seed)
            prompt = choice_prompt(observation).replace(
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
        "temperature": 0.7,
    }
    for key, value in base.items():
        if key in params:
            kwargs[key] = value
    if "max_prompt_length" in params:
        kwargs["max_prompt_length"] = 1536
    if "max_completion_length" in params:
        kwargs["max_completion_length"] = MAX_COMPLETION_LENGTH
    elif "max_length" in params:
        kwargs["max_length"] = 1536 + MAX_COMPLETION_LENGTH
    return GRPOConfig(**kwargs)


def main() -> None:
    if not Path(SFT_CHOICE_ADAPTER_PATH).exists():
        raise FileNotFoundError(
            f"Missing choice SFT adapter folder: {SFT_CHOICE_ADAPTER_PATH}. "
            "Run scripts/train_sft_choice_colab.py first."
        )

    tokenizer = AutoTokenizer.from_pretrained(SFT_CHOICE_ADAPTER_PATH, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype, device_map="auto")
    model = PeftModel.from_pretrained(base_model, SFT_CHOICE_ADAPTER_PATH, is_trainable=True)
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.print_trainable_parameters()

    dataset = build_dataset()
    print(f"GRPO choice difficulties: {DIFFICULTIES}")
    print(f"GRPO choice prompt rows: {len(dataset)}")
    args = build_grpo_args(use_bf16=use_bf16, use_fp16=use_fp16)
    trainer_kwargs = {
        "model": model,
        "args": args,
        "reward_funcs": [construx_choice_reward],
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
    print("GRPO_CHOICE_DONE")


if __name__ == "__main__":
    main()
