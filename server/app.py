from __future__ import annotations

from threading import Lock
from typing import Dict
import uuid

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from construction_safety_env.env import ConstructionSafetyEnv
from construction_safety_env.models import (
    ConstructionSafetyAction,
    ConstructionSafetyObservation,
    ConstructionSafetyReward,
    ConstructionSafetyState,
    ResetRequest,
    ResetResponse,
    SchemaResponse,
    StateResponse,
    StepRequest,
    StepResponse,
)
from construction_safety_env.tasks import list_tasks


app = FastAPI(
    title="Construx-RL",
    description="OpenEnv environment where an LLM acts as a construction site manager across permits, materials, crews, weather, OSHA safety, budget, and subcontractor negotiation.",
    version="0.2.0",
)

app.mount("/static", StaticFiles(directory="server/static"), name="static")
app.mount("/artifacts", StaticFiles(directory="demo_artifacts"), name="artifacts")

_sessions: Dict[str, ConstructionSafetyEnv] = {}
_lock = Lock()


def _get_or_create_env(session_id: str | None) -> tuple[str, ConstructionSafetyEnv]:
    with _lock:
        if session_id and session_id in _sessions:
            return session_id, _sessions[session_id]
        new_id = session_id or str(uuid.uuid4())
        env = ConstructionSafetyEnv()
        env.session_id = new_id
        _sessions[new_id] = env
        return new_id, env


def _get_session(session_id: str) -> ConstructionSafetyEnv:
    with _lock:
        env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
    return env


@app.get("/")
def root() -> FileResponse:
    return FileResponse("server/static/index.html")


@app.get("/info")
def info() -> dict:
    return {
        "name": "construx_rl",
        "tag": "openenv",
        "status": "ok",
        "tasks": [task.model_dump() for task in list_tasks()],
        "endpoints": ["/reset", "/step", "/state", "/schema", "/tasks", "/health"],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "active_sessions": len(_sessions)}


@app.get("/healthz")
def healthz() -> dict:
    return health()


@app.get("/tasks")
def tasks() -> dict:
    return {"tasks": [task.model_dump() for task in list_tasks()]}


@app.get("/schema", response_model=SchemaResponse)
def schema() -> SchemaResponse:
    return SchemaResponse(
        action_schema=ConstructionSafetyAction.model_json_schema(),
        observation_schema=ConstructionSafetyObservation.model_json_schema(),
        reward_schema=ConstructionSafetyReward.model_json_schema(),
        state_schema=ConstructionSafetyState.model_json_schema(),
    )


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest = Body(default_factory=ResetRequest)) -> ResetResponse:
    session_id, env = _get_or_create_env(request.session_id)
    observation = env.reset(difficulty=request.difficulty, task_name=request.task_name, seed=request.seed)
    return ResetResponse(session_id=session_id, observation=observation)


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest) -> StepResponse:
    env = _get_session(request.session_id)
    observation, reward, done, info = env.step(request.action)
    return StepResponse(observation=observation, reward=reward, done=done, info=info)


@app.get("/state", response_model=StateResponse)
def state(session_id: str = Query(...)) -> StateResponse:
    env = _get_session(session_id)
    return StateResponse(state=env.state())


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
