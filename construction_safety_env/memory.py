from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Iterable, List


class EpisodeMemory:
    def __init__(self, max_items: int = 8) -> None:
        self._items: Deque[str] = deque(maxlen=max_items)

    def add_episode(self, site_log: Iterable[str], reward_components: Dict[str, float]) -> None:
        failures = [line for line in site_log if "blocked" in line.lower() or "violation" in line.lower() or "failed" in line.lower()]
        if failures:
            self._items.append("; ".join(failures[-2:])[:500])
            return
        if reward_components:
            progress = reward_components.get("progress", 0.0)
            safety = reward_components.get("safety", 0.0)
            self._items.append(f"Prior episode: progress={progress:.2f}, safety={safety:.2f}; repeat dependency-first sequencing.")

    def hint(self) -> str | None:
        if not self._items:
            return None
        return " | ".join(list(self._items)[-3:])

    def clear(self) -> None:
        self._items.clear()


GLOBAL_MEMORY = EpisodeMemory()
