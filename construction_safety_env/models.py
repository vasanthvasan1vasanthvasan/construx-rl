from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class OSHAReference(BaseModel):
    citation: str
    title: str
    summary: str
    source_url: str


class FindingSubmission(BaseModel):
    hazard_label: str = Field(..., min_length=3, max_length=160)
    osha_citation: str = Field(..., min_length=3, max_length=64)
    severity: Literal["low", "medium", "high", "critical"]
    evidence: str = Field(..., min_length=8, max_length=400)
    corrective_action: str = Field(..., min_length=8, max_length=400)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class FindingFeedback(BaseModel):
    hazard_label: str
    accepted: bool
    score_delta: float = Field(ge=0.0, le=1.0)
    notes: str


class ConstructionSafetyAction(BaseModel):
    action_type: Literal["issue_finding", "submit_report"]
    finding: Optional[FindingSubmission] = None
    final_summary: Optional[str] = Field(default=None, max_length=600)


class ConstructionSafetyReward(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0)
    reason: str
    components: Dict[str, float] = Field(default_factory=dict)


class ConstructionSafetyObservation(BaseModel):
    benchmark_name: str = "construction_site_safety_inspector"
    task_name: str
    difficulty: Literal["easy", "medium", "hard"]
    inspector_role: str
    site_report: str
    objective: str
    allowed_actions: List[str]
    reference_library: List[OSHAReference]
    submitted_findings: List[FindingSubmission]
    feedback_history: List[FindingFeedback]
    step_index: int
    max_steps: int
    done: bool
    current_score: float = Field(ge=0.0, le=1.0)
    best_score: float = Field(ge=0.0, le=1.0)
    last_action_error: Optional[str] = None
    last_reward: Optional[ConstructionSafetyReward] = None
    last_info: Dict[str, Any] = Field(default_factory=dict)


class ConstructionSafetyState(BaseModel):
    session_id: str
    task_name: str
    difficulty: Literal["easy", "medium", "hard"]
    step_index: int
    max_steps: int
    done: bool
    best_score: float = Field(ge=0.0, le=1.0)
    current_score: float = Field(ge=0.0, le=1.0)
    submitted_findings: List[FindingSubmission]
    feedback_history: List[FindingFeedback]
    target_finding_ids: List[str]
    target_count: int
    last_action_error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskDescriptor(BaseModel):
    name: str
    difficulty: Literal["easy", "medium", "hard"]
    title: str
    description: str


class ResetRequest(BaseModel):
    task_name: Optional[str] = None
    seed: int = 0
    session_id: Optional[str] = None


class ResetResponse(BaseModel):
    session_id: str
    observation: ConstructionSafetyObservation


class StepRequest(BaseModel):
    session_id: str
    action: ConstructionSafetyAction


class StepResponse(BaseModel):
    observation: ConstructionSafetyObservation
    reward: ConstructionSafetyReward
    done: bool
    info: Dict[str, Any]


class StateResponse(BaseModel):
    state: ConstructionSafetyState


class SchemaResponse(BaseModel):
    action_schema: Dict[str, Any]
    observation_schema: Dict[str, Any]
    reward_schema: Dict[str, Any]
    state_schema: Dict[str, Any]
