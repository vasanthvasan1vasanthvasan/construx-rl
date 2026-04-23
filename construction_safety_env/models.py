from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


Difficulty = Literal["easy", "medium", "hard"]
Quality = Literal["cheap", "standard", "premium"]
PermitStatus = Literal["not_requested", "pending", "approved", "rejected"]
TaskState = Literal["blocked", "available", "in_progress", "done", "failed"]
CrewState = Literal["available", "assigned", "held"]


class WeatherDay(BaseModel):
    day: int
    rain_probability: float = Field(ge=0.0, le=1.0)
    wind_mph: int = Field(ge=0)
    summary: str


class TaskSnapshot(BaseModel):
    task_id: str
    name: str
    level: int
    zone: str
    status: TaskState
    prerequisites: List[str]
    remaining_days: int
    required_crew: str
    required_materials: Dict[str, int] = Field(default_factory=dict)
    required_permit: Optional[str] = None
    osha_rules: List[str] = Field(default_factory=list)
    blocked_reasons: List[str] = Field(default_factory=list)


class CrewSnapshot(BaseModel):
    crew_id: str
    crew_type: str
    status: CrewState
    assigned_task: Optional[str] = None
    daily_cost: int
    idle_cost: int


class MaterialOrder(BaseModel):
    material: str
    quantity: int = Field(gt=0)
    quality: Quality
    eta_day: int
    cost: int


class PermitSnapshot(BaseModel):
    permit_id: str
    permit_type: str
    status: PermitStatus
    requested_day: Optional[int] = None
    approval_day: Optional[int] = None


class OSHAAlert(BaseModel):
    violation_code: str
    crew_id: Optional[str] = None
    task_id: Optional[str] = None
    description: str
    incident_report_filed: bool = False


class QuoteSnapshot(BaseModel):
    quote_id: str
    subcontractor_id: str
    task_id: str
    price: int
    available_day: int
    status: Literal["open", "accepted", "rejected"]


class ConstruxAction(BaseModel):
    action_type: Literal[
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
    crew_id: Optional[str] = None
    task_id: Optional[str] = None
    reason: Optional[str] = Field(default=None, max_length=300)
    material: Optional[str] = None
    quantity: Optional[int] = Field(default=None, gt=0)
    quality: Optional[Quality] = None
    permit_type: Optional[str] = None
    permit_id: Optional[str] = None
    zone: Optional[str] = None
    violation_code: Optional[str] = None
    corrective_action: Optional[str] = Field(default=None, max_length=400)
    subcontractor_id: Optional[str] = None
    quote_id: Optional[str] = None
    counter_price: Optional[int] = Field(default=None, gt=0)


class ConstruxReward(BaseModel):
    value: float = Field(ge=-1.0, le=1.0)
    reason: str
    components: Dict[str, float] = Field(default_factory=dict)


class ConstruxObservation(BaseModel):
    benchmark_name: str = "construx_rl"
    role: str = "Construction Site Manager"
    difficulty: Difficulty
    day: int
    max_days: int
    remaining_budget: int
    starting_budget: int
    tasks: Dict[str, TaskSnapshot]
    crews: Dict[str, CrewSnapshot]
    weather_forecast: List[WeatherDay]
    inventory: Dict[str, int]
    pending_orders: List[MaterialOrder]
    permits: Dict[str, PermitSnapshot]
    inspected_zones: Dict[str, int] = Field(default_factory=dict)
    osha_alerts: List[OSHAAlert]
    subcontractor_quotes: Dict[str, QuoteSnapshot]
    site_log: List[str]
    memory_hint: Optional[str]
    allowed_actions: List[str]
    step_index: int
    max_steps: int
    done: bool
    current_score: float = Field(ge=0.0, le=1.0)
    last_action_error: Optional[str] = None
    last_reward: Optional[ConstruxReward] = None
    last_info: Dict[str, Any] = Field(default_factory=dict)


class ConstruxState(BaseModel):
    session_id: str
    difficulty: Difficulty
    day: int
    max_days: int
    step_index: int
    max_steps: int
    done: bool
    success: bool
    current_score: float = Field(ge=0.0, le=1.0)
    completed_tasks: List[str]
    failed_tasks: List[str]
    remaining_budget: int
    safety_violations: int
    missing_incident_reports: int
    reward_components: Dict[str, float] = Field(default_factory=dict)
    site_log: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ResetRequest(BaseModel):
    difficulty: Optional[Difficulty] = None
    task_name: Optional[str] = None
    seed: int = 0
    session_id: Optional[str] = None


class ResetResponse(BaseModel):
    session_id: str
    observation: ConstruxObservation


class StepRequest(BaseModel):
    session_id: str
    action: ConstruxAction


class StepResponse(BaseModel):
    observation: ConstruxObservation
    reward: ConstruxReward
    done: bool
    info: Dict[str, Any]


class StateResponse(BaseModel):
    state: ConstruxState


class SchemaResponse(BaseModel):
    action_schema: Dict[str, Any]
    observation_schema: Dict[str, Any]
    reward_schema: Dict[str, Any]
    state_schema: Dict[str, Any]


# Backward-compatible names for the existing server/client package imports.
ConstructionSafetyAction = ConstruxAction
ConstructionSafetyObservation = ConstruxObservation
ConstructionSafetyReward = ConstruxReward
ConstructionSafetyState = ConstruxState
