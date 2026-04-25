from __future__ import annotations

import inspect
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from construction_safety_env.action_choices import build_action_choices, format_action_choices, select_choice_for_action
from construction_safety_env.env import ConstructionSafetyEnv
from inference import _heuristic_action


MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
OUTPUT_DIR = os.getenv("SFT_CHOICE_OUTPUT_DIR", "construx-rl-sft-choice")
MAX_STEPS = int(os.getenv("SFT_CHOICE_MAX_STEPS", "360"))
SEEDS_PER_DIFFICULTY = int(os.getenv("SFT_CHOICE_SEEDS_PER_DIFFICULTY", "8"))
MAX_SEQ_LENGTH = int(os.getenv("SFT_CHOICE_MAX_LENGTH", "1024"))


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
        "You are the Construx-RL site manager.\n"
        "Choose exactly one numbered action from the candidate list.\n"
        "Return ONLY the number.\n\n"
        f"Day {observation.day}/{observation.max_days}; budget={observation.remaining_budget}\n"
        f"Weather: {weather}\n"
        f"Permits: {[f'{k}:{v.status}' for k, v in observation.permits.items()]}\n"
        f"Inventory: {observation.inventory}\n"
        f"Tasks:\n{tasks}\n\n"
        f"Candidate actions:\n{format_action_choices(choices)}\n"
    ), choices


def make_dataset(eos_token: str) -> Dataset:
    rows = []
    env = ConstructionSafetyEnv()

    for difficulty in ("easy", "medium", "hard"):
        for seed in range(SEEDS_PER_DIFFICULTY):
            observation = env.reset(difficulty=difficulty, seed=seed)
            while not observation.done:
                prompt, choices = choice_prompt(observation)
                heuristic_action = _heuristic_action(observation)
                choice = select_choice_for_action(choices, heuristic_action)
                rows.append({"text": prompt + str(choice.choice_id) + eos_token})
                observation, _reward, done, _info = env.step(heuristic_action)
                if done:
                    break

    env.close()
    return Dataset.from_list(rows)


def build_sft_args(use_bf16: bool, use_fp16: bool) -> SFTConfig:
    params = inspect.signature(SFTConfig).parameters
    kwargs = {"output_dir": OUTPUT_DIR}
    base = {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-4,
        "logging_steps": 5,
        "max_steps": MAX_STEPS,
        "report_to": [],
        "bf16": use_bf16,
        "fp16": use_fp16,
    }
    for key, value in base.items():
        if key in params:
            kwargs[key] = value
    if "max_length" in params:
        kwargs["max_length"] = MAX_SEQ_LENGTH
    if "packing" in params:
        kwargs["packing"] = False
    return SFTConfig(**kwargs)


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype, device_map="auto")
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    dataset = make_dataset(tokenizer.eos_token)
    print(f"SFT choice dataset rows: {len(dataset)}")
    args = build_sft_args(use_bf16=use_bf16, use_fp16=use_fp16)

    trainer_kwargs = {"model": model, "args": args, "train_dataset": dataset}
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    if "dataset_text_field" in trainer_params:
        trainer_kwargs["dataset_text_field"] = "text"

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()

    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("SFT_CHOICE_DONE")


if __name__ == "__main__":
    main()
