from __future__ import annotations

from typing import Optional

import httpx

from .models import ConstructionSafetyAction, ResetResponse, SchemaResponse, StateResponse, StepResponse


class ConstructionSafetyEnvClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        self.session_id: Optional[str] = None

    def reset(self, task_name: Optional[str] = None, seed: int = 0) -> ResetResponse:
        response = self._client.post(
            f"{self.base_url}/reset",
            json={"task_name": task_name, "seed": seed, "session_id": self.session_id},
        )
        response.raise_for_status()
        parsed = ResetResponse.model_validate(response.json())
        self.session_id = parsed.session_id
        return parsed

    def step(self, action: ConstructionSafetyAction) -> StepResponse:
        if self.session_id is None:
            raise RuntimeError("Call reset() before step().")
        response = self._client.post(
            f"{self.base_url}/step",
            json={"session_id": self.session_id, "action": action.model_dump()},
        )
        response.raise_for_status()
        return StepResponse.model_validate(response.json())

    def state(self) -> StateResponse:
        if self.session_id is None:
            raise RuntimeError("Call reset() before state().")
        response = self._client.get(f"{self.base_url}/state", params={"session_id": self.session_id})
        response.raise_for_status()
        return StateResponse.model_validate(response.json())

    def schema(self) -> SchemaResponse:
        response = self._client.get(f"{self.base_url}/schema")
        response.raise_for_status()
        return SchemaResponse.model_validate(response.json())

    def close(self) -> None:
        self._client.close()
