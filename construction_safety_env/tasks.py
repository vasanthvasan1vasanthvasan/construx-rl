from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel

from .models import Difficulty
from .scenarios import get_scenario, list_scenarios


class TaskDescriptor(BaseModel):
    name: str
    difficulty: Difficulty
    title: str
    description: str


def list_tasks() -> List[TaskDescriptor]:
    descriptors = []
    for item in list_scenarios():
        descriptors.append(
            TaskDescriptor(
                name=item["name"],
                difficulty=item["difficulty"],  # type: ignore[arg-type]
                title=f"Construx-RL {item['name'].title()} Scenario",
                description=item["description"],
            )
        )
    return descriptors


TASKS: Dict[str, TaskDescriptor] = {descriptor.name: descriptor for descriptor in list_tasks()}


def get_task(task_name: str) -> TaskDescriptor:
    if task_name not in TASKS:
        valid = ", ".join(sorted(TASKS))
        raise KeyError(f"Unknown scenario '{task_name}'. Valid scenarios: {valid}")
    return TASKS[task_name]


def scenario_summary(task_name: str) -> dict:
    descriptor = get_task(task_name)
    scenario = get_scenario(descriptor.difficulty)
    return {
        "name": descriptor.name,
        "difficulty": descriptor.difficulty,
        "task_count": len(scenario.tasks),
        "max_days": scenario.max_days,
        "starting_budget": scenario.starting_budget,
        "permit_types": scenario.permit_types,
    }
