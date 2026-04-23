from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class CrewAgent:
    crew_id: str
    crew_type: str
    behavior: str


@dataclass(frozen=True)
class SubcontractorAgent:
    subcontractor_id: str
    specialty: str
    base_markup: float
    availability_offset: int

    def quote_price(self, base_cost: int, day: int) -> int:
        scarcity = 1.0 + max(0, 10 - day) * 0.015
        return int(base_cost * self.base_markup * scarcity)

    def counter(self, proposed: int, quoted: int) -> tuple[bool, int]:
        floor = int(quoted * 0.82)
        if proposed >= floor:
            return True, proposed
        return False, int((quoted + floor) / 2)


CREW_AGENTS: Dict[str, CrewAgent] = {
    "structural": CrewAgent("structural", "structural", "Rejects work blocked by permits, missing materials, rain, or unsafe crane winds."),
    "mep": CrewAgent("mep", "mep", "Requires electrical/plumbing permits and lockout/tagout controls."),
    "finishing": CrewAgent("finishing", "finishing", "Requires roof and rough-in completion before interior finishes."),
    "admin": CrewAgent("admin", "admin", "Handles permit gates, inspections, and closeout paperwork."),
}

SUBCONTRACTOR_AGENTS: Dict[str, SubcontractorAgent] = {
    "weldco": SubcontractorAgent("weldco", "welding_stair_rails", 1.18, 2),
    "rapid_roof": SubcontractorAgent("rapid_roof", "roofing", 1.22, 1),
    "sparkrite": SubcontractorAgent("sparkrite", "mep_rough_in", 1.16, 2),
}
