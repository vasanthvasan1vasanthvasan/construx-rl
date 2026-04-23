from __future__ import annotations

from typing import Dict

from .world import ConstructionWorld


def reward_components(world: ConstructionWorld) -> Dict[str, float]:
    total_tasks = max(1, len(world.tasks))
    completed = len(world.completed_tasks())
    progress = 0.1 * completed

    if world.success:
        budget_efficiency = max(0.0, world.budget / world.scenario.starting_budget)
    else:
        budget_efficiency = 0.0

    safety = -0.5 * world.safety_violations() - 0.3 * world.missing_reports()
    if world.success and world.safety_violations() == 0:
        safety += 0.3

    schedule = 0.0
    if world.success and world.day <= 28:
        schedule = 0.5
    elif world.success and world.day <= 30:
        schedule = 0.2

    anti_hack = 0.0
    if world.last_error:
        anti_hack -= 0.08
    if world.day > world.max_days and not world.success:
        anti_hack -= 0.2
    if world.budget <= 0:
        anti_hack -= 0.4

    normalized_progress = completed / total_tasks
    return {
        "progress": round(progress, 4),
        "normalized_progress": round(normalized_progress, 4),
        "budget_efficiency": round(budget_efficiency, 4),
        "safety": round(safety, 4),
        "schedule": round(schedule, 4),
        "anti_hack": round(anti_hack, 4),
    }


def score_world(world: ConstructionWorld) -> float:
    components = reward_components(world)
    raw = (
        0.45 * components["normalized_progress"]
        + 0.20 * components["budget_efficiency"]
        + 0.20 * max(0.0, min(1.0, 0.7 + components["safety"]))
        + 0.15 * components["schedule"]
        + components["anti_hack"]
    )
    return round(max(0.0, min(1.0, raw)), 4)


def step_reward(before_score: float, after_score: float, world: ConstructionWorld, info: Dict[str, object]) -> tuple[float, str, Dict[str, float]]:
    components = reward_components(world)
    delta = after_score - before_score
    dense_progress = 0.1 * float(info.get("new_tasks_completed", 0))
    value = delta + dense_progress
    if world.last_error:
        value -= 0.05
    if world.safety_violations():
        value -= 0.02 * world.safety_violations()
    value = round(max(-1.0, min(1.0, value)), 4)

    if world.success:
        reason = "Project completed; reward includes progress, budget, safety, and schedule."
    elif world.last_error:
        reason = f"Action rejected: {world.last_error}"
    elif dense_progress > 0:
        reason = "Task progress completed in valid dependency order."
    else:
        reason = "Management action accepted; reward reflects updated project state."
    return value, reason, components
