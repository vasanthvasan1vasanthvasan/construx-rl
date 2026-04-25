"""
Minimal Colab-oriented GRPO training skeleton for Construx-RL.

Install cells:
    !pip install -U openenv-core trl unsloth transformers accelerate datasets
    !pip install git+https://huggingface.co/spaces/YOUR_ORG/construx-rl

This file is intentionally small: the environment is the source of reward truth,
and the trainer only needs prompt sampling plus a verifier-style reward function.
"""

from __future__ import annotations

import json
from typing import List

from unsloth import FastLanguageModel
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from construction_safety_env.env import ConstructionSafetyEnv
from construction_safety_env.models import ConstruxAction


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_SEQ_LENGTH = 2048


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
        "Return one JSON action for Construx-RL. Valid action_type values: "
        "assign_crew, hold_crew, order_material, check_inventory, check_weather, request_permit, "
        "check_permit_status, file_incident_report, request_inspection, request_quote, accept_quote, negotiate.\n"
        "Optimize dependency order, permits, material lead times, OSHA safety, budget, and schedule.\n\n"
        f"Day {observation.day}/{observation.max_days}; budget={observation.remaining_budget}; permits={permits}\n"
        f"Weather: {weather}\nInventory: {observation.inventory}\nTasks:\n{tasks}\n"
        f"Alerts: {[alert.model_dump() for alert in observation.osha_alerts]}\n"
    )


def extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found.")
    return json.loads(text[start : end + 1])


def construx_reward(completions: List[str], prompts: List[str], **_: object) -> List[float]:
    rewards: List[float] = []
    for completion in completions:
        env = ConstructionSafetyEnv()
        observation = env.reset(difficulty="easy", seed=0)
        total = 0.0
        try:
            for _ in range(8):
                action = ConstruxAction.model_validate(extract_json(completion))
                observation, reward, done, _info = env.step(action)
                total += reward.value
                if done:
                    break
            total += env.state().current_score
        except Exception:
            total -= 0.25
        rewards.append(float(total))
        env.close()
    return rewards


def main() -> None:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    env = ConstructionSafetyEnv()
    prompts = []
    for difficulty in ("easy", "medium", "hard"):
        observation = env.reset(difficulty=difficulty, seed=0)
        prompts.append({"prompt": observation_prompt(observation)})
    dataset = Dataset.from_list(prompts * 64)

    args = GRPOConfig(
        output_dir="construx-rl-grpo",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=2,
        max_prompt_length=1536,
        max_completion_length=384,
        learning_rate=5e-6,
        logging_steps=5,
        max_steps=20,
        bf16=False,
        fp16=True,
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[construx_reward],
        args=args,
        train_dataset=dataset,
    )
    trainer.train()
    model.save_pretrained("construx-rl-lora")
    tokenizer.save_pretrained("construx-rl-lora")


if __name__ == "__main__":
    main()
