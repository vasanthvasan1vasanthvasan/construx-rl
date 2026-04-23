from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .models import Difficulty


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    name: str
    level: int
    zone: str
    duration: int
    required_crew: str
    prerequisites: List[str] = field(default_factory=list)
    required_materials: Dict[str, int] = field(default_factory=dict)
    required_permit: Optional[str] = None
    osha_rules: List[str] = field(default_factory=list)
    outdoor: bool = False
    crane: bool = False
    height_ft: int = 0
    excavation_depth_ft: int = 0
    scaffold: bool = False
    welding: bool = False
    electrical: bool = False


@dataclass(frozen=True)
class CrewSpec:
    crew_id: str
    crew_type: str
    daily_cost: int
    idle_cost: int


@dataclass(frozen=True)
class ScenarioSpec:
    difficulty: Difficulty
    max_days: int
    max_steps: int
    starting_budget: int
    tasks: List[TaskSpec]
    crews: List[CrewSpec]
    initial_inventory: Dict[str, int]
    permit_types: List[str]
    subcontractors: Dict[str, List[str]]


ALL_TASKS: List[TaskSpec] = [
    TaskSpec("site_survey", "Site survey", 0, "site", 1, "structural"),
    TaskSpec("building_permit_gate", "Building permit gate", 0, "office", 1, "admin", required_permit="building"),
    TaskSpec(
        "excavation",
        "Excavation",
        1,
        "zone_a",
        1,
        "structural",
        ["site_survey", "building_permit_gate"],
        osha_rules=["OSHA 1926.652", "OSHA 1926.100"],
        outdoor=True,
        excavation_depth_ft=7,
    ),
    TaskSpec(
        "foundation_pour",
        "Foundation pour",
        2,
        "zone_a",
        1,
        "structural",
        ["excavation"],
        {"concrete": 10},
        "building",
        ["OSHA 1926.32", "OSHA 1926.100"],
        outdoor=True,
    ),
    TaskSpec("foundation_cure", "Foundation cure", 2, "zone_a", 1, "structural", ["foundation_pour"]),
    TaskSpec(
        "structural_framing",
        "Structural framing",
        3,
        "zone_b",
        2,
        "structural",
        ["foundation_cure"],
        {"steel": 8, "lumber": 8},
        "building",
        ["OSHA 1926.502", "OSHA 1926.550", "OSHA 1926.100"],
        outdoor=True,
        crane=True,
        height_ft=14,
    ),
    TaskSpec(
        "roofing",
        "Roofing",
        4,
        "roof",
        2,
        "structural",
        ["structural_framing"],
        {"roofing": 6},
        "building",
        ["OSHA 1926.502", "OSHA 1926.451", "OSHA 1926.100"],
        outdoor=True,
        height_ft=18,
        scaffold=True,
    ),
    TaskSpec(
        "mep_rough_in",
        "MEP rough-in",
        4,
        "interior",
        2,
        "mep",
        ["structural_framing"],
        {"wire": 5, "pipe": 5},
        "electrical",
        ["OSHA 1910.147", "OSHA 1926.100"],
        electrical=True,
    ),
    TaskSpec("plumbing_pressure_test", "Plumbing pressure test", 5, "interior", 1, "mep", ["mep_rough_in"], {"pipe": 2}, "plumbing"),
    TaskSpec("electrical_inspection", "Electrical inspection", 5, "interior", 1, "mep", ["mep_rough_in"], required_permit="electrical"),
    TaskSpec("insulation", "Insulation", 5, "interior", 1, "finishing", ["roofing", "mep_rough_in"], {"insulation": 6}),
    TaskSpec("drywall", "Drywall", 5, "interior", 2, "finishing", ["insulation", "electrical_inspection"], {"drywall": 8}),
    TaskSpec(
        "welding_stair_rails",
        "Weld stair rails",
        6,
        "stairwell",
        1,
        "subcontractor",
        ["drywall"],
        {"steel": 2},
        "building",
        ["OSHA 1926.150", "OSHA 1926.100"],
        welding=True,
    ),
    TaskSpec("finishing", "Finishing", 6, "interior", 2, "finishing", ["drywall"], {"paint": 6}),
    TaskSpec("final_inspection", "Final inspection", 6, "site", 1, "admin", ["finishing", "roofing", "plumbing_pressure_test", "welding_stair_rails"], required_permit="building"),
]


CREWS = [
    CrewSpec("structural", "structural", 2600, 220),
    CrewSpec("mep", "mep", 2300, 190),
    CrewSpec("finishing", "finishing", 1900, 160),
    CrewSpec("admin", "admin", 900, 80),
]


def get_scenario(difficulty: Difficulty) -> ScenarioSpec:
    if difficulty == "easy":
        task_ids = {"site_survey", "building_permit_gate", "excavation", "foundation_pour", "foundation_cure"}
        budget = 52000
        permits = ["building"]
        inventory = {"concrete": 0}
        max_days = 12
        max_steps = 40
    elif difficulty == "medium":
        task_ids = {
            "site_survey",
            "building_permit_gate",
            "excavation",
            "foundation_pour",
            "foundation_cure",
            "structural_framing",
            "roofing",
            "mep_rough_in",
            "electrical_inspection",
            "drywall",
        }
        budget = 145000
        permits = ["building", "electrical", "plumbing"]
        inventory = {"concrete": 0, "steel": 0, "lumber": 0, "roofing": 0, "wire": 0, "pipe": 0, "drywall": 0}
        max_days = 22
        max_steps = 70
    else:
        task_ids = {task.task_id for task in ALL_TASKS}
        budget = 230000
        permits = ["building", "electrical", "plumbing"]
        inventory = {
            "concrete": 0,
            "steel": 0,
            "lumber": 0,
            "roofing": 0,
            "wire": 0,
            "pipe": 0,
            "insulation": 0,
            "drywall": 0,
            "paint": 0,
        }
        max_days = 30
        max_steps = 90

    tasks = [task for task in ALL_TASKS if task.task_id in task_ids]
    crews = [crew for crew in CREWS if crew.crew_type in {task.required_crew for task in tasks} or crew.crew_type == "admin"]
    return ScenarioSpec(
        difficulty=difficulty,
        max_days=max_days,
        max_steps=max_steps,
        starting_budget=budget,
        tasks=tasks,
        crews=crews,
        initial_inventory=inventory,
        permit_types=permits,
        subcontractors={"weldco": ["welding_stair_rails"], "rapid_roof": ["roofing"], "sparkrite": ["mep_rough_in"]},
    )


def list_scenarios() -> List[Dict[str, str]]:
    return [
        {"name": "easy", "difficulty": "easy", "description": "Single zone, five tasks, stable weather, one permit."},
        {"name": "medium", "difficulty": "medium", "description": "Ten tasks with weather disruption, three permits, and one subcontractor option."},
        {"name": "hard", "difficulty": "hard", "description": "Full 30-day, 15-task construction project with OSHA, budget, materials, permits, and negotiations."},
    ]
