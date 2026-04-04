from __future__ import annotations

import random
import uuid
from typing import Any, Dict, Optional, Tuple

from .grader import grade_task
from .models import (
    ConstructionSafetyAction,
    ConstructionSafetyObservation,
    ConstructionSafetyReward,
    ConstructionSafetyState,
    FindingFeedback,
)
from .tasks import TASKS, get_task


class ConstructionSafetyEnv:
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self.session_id = str(uuid.uuid4())
        self._rng = random.Random(0)
        self._task = None
        self._submitted_findings = []
        self._feedback_history = []
        self._done = False
        self._step_index = 0
        self._best_score = 0.0
        self._current_score = 0.0
        self._last_action_error: Optional[str] = None
        self._last_reward: Optional[ConstructionSafetyReward] = None
        self._last_info: Dict[str, Any] = {}
        self._submitted_final = False

    def reset(self, task_name: Optional[str] = None, seed: int = 0) -> ConstructionSafetyObservation:
        self._rng = random.Random(seed)
        if task_name is None:
            task_name = self._rng.choice(sorted(TASKS))
        self._task = get_task(task_name)
        self._submitted_findings = []
        self._feedback_history = []
        self._done = False
        self._step_index = 0
        self._best_score = 0.0
        self._current_score = 0.0
        self._last_action_error = None
        self._last_reward = None
        self._last_info = {"task_loaded": True}
        self._submitted_final = False
        return self._build_observation()

    def state(self) -> ConstructionSafetyState:
        self._ensure_task_loaded()
        assert self._task is not None
        return ConstructionSafetyState(
            session_id=self.session_id,
            task_name=self._task.descriptor.name,
            difficulty=self._task.descriptor.difficulty,
            step_index=self._step_index,
            max_steps=self._task.max_steps,
            done=self._done,
            best_score=self._best_score,
            current_score=self._current_score,
            submitted_findings=list(self._submitted_findings),
            feedback_history=list(self._feedback_history),
            target_finding_ids=[finding.finding_id for finding in self._task.target_findings],
            target_count=len(self._task.target_findings),
            last_action_error=self._last_action_error,
            metadata={"grading_notes": self._task.grading_notes},
        )

    def step(
        self, action: ConstructionSafetyAction
    ) -> Tuple[ConstructionSafetyObservation, ConstructionSafetyReward, bool, Dict[str, Any]]:
        self._ensure_task_loaded()
        assert self._task is not None

        if self._done:
            reward = ConstructionSafetyReward(value=0.0, reason="Episode already finished.", components={"stale_action": 0.0})
            info = {"error": "Episode already finished."}
            self._last_action_error = info["error"]
            self._last_reward = reward
            self._last_info = info
            return self._build_observation(), reward, True, info

        self._step_index += 1
        previous_best = self._best_score
        self._last_action_error = None

        if action.action_type == "issue_finding":
            if action.finding is None:
                reward = ConstructionSafetyReward(value=0.0, reason="Missing finding payload.", components={"validation": 0.0})
                info = {"error": "issue_finding requires a finding payload"}
                self._last_action_error = info["error"]
            else:
                self._submitted_findings.append(action.finding)
                evaluation = grade_task(self._task, self._submitted_findings, steps_used=self._step_index, submitted_final=False)
                self._current_score = float(evaluation["score"])
                self._best_score = max(self._best_score, self._current_score)
                score_delta = max(0.0, round(self._best_score - previous_best, 4))
                self._feedback_history.append(
                    FindingFeedback(
                        hazard_label=action.finding.hazard_label,
                        accepted=score_delta > 0.0,
                        score_delta=score_delta,
                        notes="Finding improved task score." if score_delta > 0.0 else "Finding did not improve score; likely duplicate, weak evidence, or incorrect citation.",
                    )
                )
                reward = ConstructionSafetyReward(
                    value=score_delta,
                    reason="Incremental progress reward based on deterministic grader gain.",
                    components={"score_delta": score_delta, "current_score": self._current_score, "best_score": self._best_score},
                )
                info = evaluation
        elif action.action_type == "submit_report":
            self._submitted_final = True
            evaluation = grade_task(self._task, self._submitted_findings, steps_used=self._step_index, submitted_final=True)
            self._current_score = float(evaluation["score"])
            self._best_score = max(self._best_score, self._current_score)
            reward = ConstructionSafetyReward(
                value=max(0.0, round(self._current_score - previous_best, 4)),
                reason="Final report reward after deterministic grading.",
                components={
                    "final_score": self._current_score,
                    "matched_count": float(evaluation["matched_count"]),
                    "target_count": float(evaluation["target_count"]),
                },
            )
            info = evaluation
            self._done = True
        else:
            reward = ConstructionSafetyReward(value=0.0, reason="Unsupported action.", components={"validation": 0.0})
            info = {"error": f"Unsupported action type: {action.action_type}"}
            self._last_action_error = info["error"]

        if self._step_index >= self._task.max_steps and not self._done:
            evaluation = grade_task(self._task, self._submitted_findings, steps_used=self._step_index, submitted_final=self._submitted_final)
            self._current_score = float(evaluation["score"])
            self._best_score = max(self._best_score, self._current_score)
            self._done = True
            info = {**evaluation, "auto_terminated": True, "error": self._last_action_error}
            reward = ConstructionSafetyReward(
                value=max(reward.value, max(0.0, round(self._current_score - previous_best, 4))),
                reason="Episode reached maximum steps.",
                components={**reward.components, "final_score": self._current_score},
            )

        self._last_reward = reward
        self._last_info = info
        return self._build_observation(), reward, self._done, info

    def close(self) -> None:
        return None

    def _ensure_task_loaded(self) -> None:
        if self._task is None:
            self.reset(task_name="easy_roof_fall_protection", seed=0)

    def _build_observation(self) -> ConstructionSafetyObservation:
        self._ensure_task_loaded()
        assert self._task is not None
        return ConstructionSafetyObservation(
            task_name=self._task.descriptor.name,
            difficulty=self._task.descriptor.difficulty,
            inspector_role=self._task.inspector_role,
            site_report=self._task.site_report,
            objective=self._task.objective,
            allowed_actions=["issue_finding", "submit_report"],
            reference_library=self._task.references,
            submitted_findings=list(self._submitted_findings),
            feedback_history=list(self._feedback_history),
            step_index=self._step_index,
            max_steps=self._task.max_steps,
            done=self._done,
            current_score=self._current_score,
            best_score=self._best_score,
            last_action_error=self._last_action_error,
            last_reward=self._last_reward,
            last_info=dict(self._last_info),
        )
