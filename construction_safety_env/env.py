from __future__ import annotations

import uuid
from typing import Any, Dict, Optional, Tuple

from .curriculum import GLOBAL_CURRICULUM
from .grader import score_world, step_reward
from .memory import GLOBAL_MEMORY
from .models import ConstruxAction, ConstruxObservation, ConstruxReward, ConstruxState, Difficulty
from .world import ConstructionWorld


ALLOWED_ACTIONS = [
    "assign_crew",
    "hold_crew",
    "order_material",
    "check_inventory",
    "check_weather",
    "request_permit",
    "check_permit_status",
    "file_incident_report",
    "request_inspection",
    "request_quote",
    "accept_quote",
    "negotiate",
]


class ConstructionSafetyEnv:
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self.session_id = str(uuid.uuid4())
        self._world: Optional[ConstructionWorld] = None
        self._last_reward: Optional[ConstruxReward] = None
        self._last_info: Dict[str, Any] = {}
        self._current_score = 0.0
        self._reward_components: Dict[str, float] = {}

    def reset(
        self,
        difficulty: Optional[Difficulty] = None,
        task_name: Optional[str] = None,
        seed: int = 0,
    ) -> ConstruxObservation:
        requested = difficulty or self._difficulty_from_task_name(task_name)
        chosen = GLOBAL_CURRICULUM.choose(requested)
        self._world = ConstructionWorld(chosen, seed=seed, memory_hint=GLOBAL_MEMORY.hint())
        self._current_score = score_world(self._world)
        self._reward_components = {}
        self._last_reward = None
        self._last_info = {"scenario_loaded": chosen}
        return self._build_observation()

    def state(self) -> ConstruxState:
        world = self._ensure_world()
        return ConstruxState(
            session_id=self.session_id,
            difficulty=world.difficulty,
            day=world.day,
            max_days=world.max_days,
            step_index=world.step_index,
            max_steps=world.max_steps,
            done=world.done,
            success=world.success,
            current_score=self._current_score,
            completed_tasks=world.completed_tasks(),
            failed_tasks=world.failed_tasks(),
            remaining_budget=world.budget,
            safety_violations=world.safety_violations(),
            missing_incident_reports=world.missing_reports(),
            reward_components=dict(self._reward_components),
            site_log=list(world.site_log),
            metadata={
                "curriculum_difficulty": GLOBAL_CURRICULUM.difficulty,
                "task_count": len(world.tasks),
                "open_quotes": list(world.quotes),
            },
        )

    def step(self, action: ConstruxAction) -> Tuple[ConstruxObservation, ConstruxReward, bool, Dict[str, Any]]:
        world = self._ensure_world()
        if world.done:
            reward = ConstruxReward(value=0.0, reason="Episode already finished.", components={"stale_action": 0.0})
            info = {"error": "Episode already finished."}
            self._last_reward = reward
            self._last_info = info
            return self._build_observation(), reward, True, info

        before_score = self._current_score
        payload = action.model_dump(exclude={"action_type"}, exclude_none=True)
        info = world.apply(action.action_type, payload)
        after_score = score_world(world)
        value, reason, components = step_reward(before_score, after_score, world, info)
        self._current_score = after_score
        self._reward_components = components

        reward = ConstruxReward(value=value, reason=reason, components=components)
        if world.done:
            GLOBAL_CURRICULUM.record(after_score)
            GLOBAL_MEMORY.add_episode(world.site_log, components)
        self._last_reward = reward
        self._last_info = info
        return self._build_observation(), reward, world.done, info

    def close(self) -> None:
        return None

    def _ensure_world(self) -> ConstructionWorld:
        if self._world is None:
            self.reset(difficulty="easy", seed=0)
        assert self._world is not None
        return self._world

    def _build_observation(self) -> ConstruxObservation:
        world = self._ensure_world()
        return ConstruxObservation(
            difficulty=world.difficulty,
            day=world.day,
            max_days=world.max_days,
            remaining_budget=world.budget,
            starting_budget=world.scenario.starting_budget,
            tasks=world.task_snapshots(),
            crews=world.crew_snapshots(),
            weather_forecast=world.weather_forecast(),
            inventory=dict(world.inventory),
            pending_orders=list(world.pending_orders),
            permits=dict(world.permits),
            inspected_zones=dict(world.inspected_zones),
            osha_alerts=list(world.alerts),
            subcontractor_quotes=dict(world.quotes),
            site_log=list(world.site_log[-12:]),
            memory_hint=world.memory_hint,
            allowed_actions=ALLOWED_ACTIONS,
            step_index=world.step_index,
            max_steps=world.max_steps,
            done=world.done,
            current_score=self._current_score,
            last_action_error=world.last_error,
            last_reward=self._last_reward,
            last_info=dict(self._last_info),
        )

    @staticmethod
    def _difficulty_from_task_name(task_name: Optional[str]) -> Optional[Difficulty]:
        if task_name in {"easy", "medium", "hard"}:
            return task_name  # type: ignore[return-value]
        return None
