from __future__ import annotations

import json
import os
from typing import List, Optional
from urllib.parse import urlparse

from openai import OpenAI

from construction_safety_env.client import ConstructionSafetyEnvClient
from construction_safety_env.env import ConstructionSafetyEnv
from construction_safety_env.models import ConstruxAction, Difficulty


API_BASE_URL = os.getenv("API_BASE_URL", "local")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
BENCHMARK = "construx_rl"
MAX_TOKENS = 500
TEMPERATURE = 0.0


def _is_http_url(url: Optional[str]) -> bool:
    return bool(url and urlparse(url).scheme in {"http", "https"})


def _format_action(action: ConstruxAction) -> str:
    payload = action.model_dump(exclude_none=True)
    action_type = payload.pop("action_type")
    args = ", ".join(f"{key}={value!r}" for key, value in payload.items())
    return f"{action_type}({args})"


def _print_start(difficulty: str) -> None:
    print(f"[START] env={BENCHMARK} difficulty={difficulty} model={MODEL_NAME}")


def _print_step(step: int, day: int, action: ConstruxAction, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} day={day} action={_format_action(action)} reward={reward:.3f} "
        f"done={'true' if done else 'false'} error={error if error else 'null'}"
    )


def _print_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    reward_text = ",".join(f"{reward:.3f}" for reward in rewards)
    print(f"[END] success={'true' if success else 'false'} steps={steps} score={score:.3f} rewards={reward_text}")


class _LocalEnvAdapter:
    def __init__(self, difficulty: Difficulty) -> None:
        self._env = ConstructionSafetyEnv()
        self._difficulty = difficulty

    def reset(self):
        class Result:
            def __init__(self, observation) -> None:
                self.observation = observation

        return Result(self._env.reset(difficulty=self._difficulty, seed=0))

    def step(self, action: ConstruxAction):
        observation, reward, done, info = self._env.step(action)

        class Result:
            def __init__(self, observation, reward, done, info) -> None:
                self.observation = observation
                self.reward = reward
                self.done = done
                self.info = info

        return Result(observation, reward, done, info)

    def state(self):
        class Result:
            def __init__(self, state) -> None:
                self.state = state

        return Result(self._env.state())

    def close(self) -> None:
        self._env.close()


class _HttpEnvAdapter:
    def __init__(self, difficulty: Difficulty, base_url: str) -> None:
        self._client = ConstructionSafetyEnvClient(base_url=base_url)
        self._difficulty = difficulty

    def reset(self):
        return self._client.reset(difficulty=self._difficulty, seed=0)

    def step(self, action: ConstruxAction):
        return self._client.step(action)

    def state(self):
        return self._client.state()

    def close(self) -> None:
        self._client.close()


def _make_env(difficulty: Difficulty):
    if _is_http_url(API_BASE_URL):
        return _HttpEnvAdapter(difficulty=difficulty, base_url=API_BASE_URL)
    return _LocalEnvAdapter(difficulty)


def _build_prompt(observation) -> str:
    tasks = "\n".join(
        f"- {task.task_id}: {task.status}, needs crew={task.required_crew}, blocked={task.blocked_reasons}"
        for task in observation.tasks.values()
    )
    permits = ", ".join(f"{name}:{permit.status}" for name, permit in observation.permits.items())
    weather = ", ".join(
        f"day {item.day} rain={item.rain_probability} wind={item.wind_mph}" for item in observation.weather_forecast
    )
    alerts = "\n".join(f"- {alert.violation_code}: {alert.description}" for alert in observation.osha_alerts) or "- none"
    return (
        "You are the Construx-RL construction site manager. Return JSON only matching this schema:\n"
        '{"action_type":"assign_crew|hold_crew|order_material|check_inventory|check_weather|request_permit|check_permit_status|file_incident_report|request_inspection|request_quote|accept_quote|negotiate", "...":"..."}\n'
        "Plan dependency-first: permits early, materials three days early, inspect hazardous zones before risky work, avoid rain/crane wind, file incident reports.\n\n"
        f"Day {observation.day}/{observation.max_days}, budget={observation.remaining_budget}, permits={permits}\n"
        f"Weather: {weather}\n"
        f"Inventory: {observation.inventory}\n"
        f"Tasks:\n{tasks}\n"
        f"OSHA alerts:\n{alerts}\n"
        f"Recent site log: {observation.site_log[-5:]}\n"
    )


def _extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found: {text}")
    return json.loads(text[start : end + 1])


def _first_available_task(observation, crew_type: str):
    for task in observation.tasks.values():
        if task.status == "available" and task.required_crew == crew_type:
            return task
    return None


def _first_blocked_material_need(observation):
    for task in observation.tasks.values():
        if task.status in {"done", "in_progress"}:
            continue
        for material, qty in task.required_materials.items():
            pending = sum(order.quantity for order in observation.pending_orders if order.material == material)
            if observation.inventory.get(material, 0) + pending < qty:
                return material, qty
    return None


def _zone_already_inspected(observation, zone: str) -> bool:
    return zone in observation.inspected_zones


def _heuristic_action(observation) -> ConstruxAction:
    for alert in observation.osha_alerts:
        if not alert.incident_report_filed:
            fixes = {
                "OSHA 1926.502": "Install guardrails or fall arrest before elevated work resumes.",
                "OSHA 1926.652": "Use shoring or a trench box before excavation entry.",
                "OSHA 1926.550": "Clear and barricade the crane swing radius and wait for safe wind.",
                "OSHA 1910.147": "Apply lockout/tagout and de-energize electrical systems.",
                "OSHA 1926.451": "Inspect and tag scaffolding before the shift.",
                "OSHA 1926.150": "Place a fire extinguisher within 100 feet before welding.",
            }
            return ConstruxAction(
                action_type="file_incident_report",
                violation_code=alert.violation_code,
                crew_id=alert.crew_id,
                corrective_action=fixes.get(alert.violation_code, "Correct the cited hazard and retrain the crew."),
            )

    for permit_type, permit in observation.permits.items():
        if permit.status == "not_requested":
            return ConstruxAction(action_type="request_permit", permit_type=permit_type)

    need = _first_blocked_material_need(observation)
    if need:
        material, qty = need
        return ConstruxAction(action_type="order_material", material=material, quantity=qty, quality="standard")

    for task in observation.tasks.values():
        if task.status == "available" and task.osha_rules and task.zone not in {"office", "site"} and not _zone_already_inspected(observation, task.zone):
            return ConstruxAction(action_type="request_inspection", zone=task.zone)

    for task in observation.tasks.values():
        if task.status == "available" and task.required_crew == "subcontractor":
            open_quotes = [quote for quote in observation.subcontractor_quotes.values() if quote.task_id == task.task_id and quote.status == "open"]
            if not open_quotes:
                subcontractor = "weldco" if task.task_id == "welding_stair_rails" else "rapid_roof"
                return ConstruxAction(action_type="request_quote", subcontractor_id=subcontractor, task_id=task.task_id)
            quote = open_quotes[0]
            if quote.available_day <= observation.day:
                return ConstruxAction(action_type="accept_quote", quote_id=quote.quote_id)
            for crew in observation.crews.values():
                if crew.status in {"available", "held"}:
                    return ConstruxAction(action_type="hold_crew", crew_id=crew.crew_id, reason="Waiting for subcontractor availability.")

    if observation.weather_forecast and (
        observation.weather_forecast[0].rain_probability >= 0.35 or observation.weather_forecast[0].wind_mph > 22
    ):
        checked_today = any(f"Day {observation.day}: weather checked" in line for line in observation.site_log)
        if not checked_today:
            return ConstruxAction(action_type="check_weather")
        for crew in observation.crews.values():
            if crew.status in {"available", "held"}:
                return ConstruxAction(action_type="hold_crew", crew_id=crew.crew_id, reason="Waiting for safer weather.")

    for crew in observation.crews.values():
        if crew.status == "assigned" and crew.assigned_task:
            return ConstruxAction(action_type="assign_crew", crew_id=crew.crew_id, task_id=crew.assigned_task)

    for crew in observation.crews.values():
        task = _first_available_task(observation, crew.crew_type)
        if task:
            return ConstruxAction(action_type="assign_crew", crew_id=crew.crew_id, task_id=task.task_id)

    for crew in observation.crews.values():
        if crew.status == "available":
            return ConstruxAction(action_type="hold_crew", crew_id=crew.crew_id, reason="Waiting for dependency, permit, material, weather, or subcontractor availability.")

    return ConstruxAction(action_type="check_inventory")


def _llm_action(client: OpenAI, observation) -> ConstruxAction:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": "You are a careful construction site manager. Return JSON only."},
            {"role": "user", "content": _build_prompt(observation)},
        ],
    )
    payload = _extract_json(response.choices[0].message.content or "")
    return ConstruxAction.model_validate(payload)


def run_episode(difficulty: Difficulty, client: Optional[OpenAI]) -> dict:
    env = _make_env(difficulty)
    observation = env.reset().observation
    rewards: List[float] = []
    _print_start(difficulty)
    try:
        while not observation.done:
            try:
                action = _llm_action(client, observation) if client is not None else _heuristic_action(observation)
            except Exception:
                action = _heuristic_action(observation)
            result = env.step(action)
            observation = result.observation
            rewards.append(result.reward.value)
            _print_step(observation.step_index, observation.day, action, result.reward.value, result.done, observation.last_action_error)
            if result.done:
                break
    finally:
        state = env.state().state
        _print_end(state.success, len(rewards), state.current_score, rewards)
        env.close()
    return {"score": state.current_score, "steps": len(rewards), "success": state.success}


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if HF_TOKEN and _is_http_url(API_BASE_URL) else None
    for difficulty in ("easy", "medium", "hard"):
        run_episode(difficulty, client)


if __name__ == "__main__":
    main()
