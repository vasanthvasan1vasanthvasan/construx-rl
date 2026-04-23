from __future__ import annotations

from collections import deque
from typing import Deque

from .models import Difficulty


class Curriculum:
    def __init__(self) -> None:
        self._recent_rewards: Deque[float] = deque(maxlen=5)
        self._difficulty: Difficulty = "easy"

    @property
    def difficulty(self) -> Difficulty:
        return self._difficulty

    def record(self, reward: float) -> None:
        self._recent_rewards.append(reward)
        if len(self._recent_rewards) < 3:
            return
        avg = sum(self._recent_rewards) / len(self._recent_rewards)
        if self._difficulty == "easy" and avg >= 0.50:
            self._difficulty = "medium"
        elif self._difficulty == "medium" and avg >= 0.75:
            self._difficulty = "hard"

    def choose(self, requested: Difficulty | None) -> Difficulty:
        return requested or self._difficulty


GLOBAL_CURRICULUM = Curriculum()
