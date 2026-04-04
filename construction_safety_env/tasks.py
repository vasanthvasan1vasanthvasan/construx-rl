from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, Field

from .models import OSHAReference, TaskDescriptor


class TargetFinding(BaseModel):
    finding_id: str
    label: str
    label_aliases: List[str]
    osha_citation: str
    citation_title: str
    severity: Literal["low", "medium", "high", "critical"]
    evidence_keywords: List[str]
    corrective_keywords: List[str]


class InspectionTask(BaseModel):
    descriptor: TaskDescriptor
    objective: str
    inspector_role: str
    max_steps: int = Field(default=6, ge=3, le=12)
    site_report: str
    references: List[OSHAReference]
    target_findings: List[TargetFinding]
    grading_notes: str


REFERENCE_LIBRARY: List[OSHAReference] = [
    OSHAReference(
        citation="29 CFR 1926.501(b)(1)",
        title="Unprotected sides and edges",
        summary="Workers exposed to falls of 6 feet or more at an unprotected edge need guardrails, nets, or personal fall arrest.",
        source_url="https://www.osha.gov/laws-regs/regulations/standardnumber/1926/1926.501",
    ),
    OSHAReference(
        citation="29 CFR 1926.1053(b)(1)",
        title="Portable ladder side rails",
        summary="Portable ladders used for upper landing access must extend at least 3 feet above the landing surface or provide equivalent grasping support.",
        source_url="https://www.osha.gov/laws-regs/regulations/standardnumber/1926/1926.1053",
    ),
    OSHAReference(
        citation="29 CFR 1926.1053(b)(6)",
        title="Ladder footing and stability",
        summary="Portable ladders may be used only on stable and level surfaces unless secured to prevent accidental displacement.",
        source_url="https://www.osha.gov/laws-regs/regulations/standardnumber/1926/1926.1053",
    ),
    OSHAReference(
        citation="29 CFR 1926.652(a)(1)",
        title="Protective systems in excavations",
        summary="Employees in excavations must be protected from cave-ins by a protective system unless the excavation is made entirely in stable rock or is less than 5 feet deep with no cave-in indication.",
        source_url="https://www.osha.gov/laws-regs/regulations/standardnumber/1926/1926.652",
    ),
    OSHAReference(
        citation="29 CFR 1926.651(c)(2)",
        title="Means of egress in trenches",
        summary="A stairway, ladder, ramp, or other safe egress is required in trench excavations 4 feet or more deep so workers do not travel more than 25 feet laterally.",
        source_url="https://www.osha.gov/laws-regs/regulations/standardnumber/1926/1926.651",
    ),
    OSHAReference(
        citation="29 CFR 1926.651(j)(2)",
        title="Spoil piles kept back from edge",
        summary="Excavated material and equipment must be kept at least 2 feet from the edge of excavations, or restrained from falling in.",
        source_url="https://www.osha.gov/laws-regs/regulations/standardnumber/1926/1926.651",
    ),
    OSHAReference(
        citation="29 CFR 1926.451(g)(1)",
        title="Fall protection on scaffolds",
        summary="Each employee on a scaffold more than 10 feet above a lower level must be protected from falling.",
        source_url="https://www.osha.gov/laws-regs/regulations/standardnumber/1926/1926.451",
    ),
    OSHAReference(
        citation="29 CFR 1926.451(h)(1)",
        title="Falling object protection on scaffolds",
        summary="Employees on scaffolds must be protected from tools, materials, and equipment falling from above.",
        source_url="https://www.osha.gov/laws-regs/regulations/standardnumber/1926/1926.451",
    ),
    OSHAReference(
        citation="29 CFR 1926.501(b)(4)(i)",
        title="Floor holes",
        summary="Employees on walking or working surfaces must be protected from falling through holes by covers, guardrails, nets, or personal fall arrest.",
        source_url="https://www.osha.gov/laws-regs/regulations/standardnumber/1926/1926.501",
    ),
]


TASKS: Dict[str, InspectionTask] = {
    "easy_roof_fall_protection": InspectionTask(
        descriptor=TaskDescriptor(
            name="easy_roof_fall_protection",
            difficulty="easy",
            title="Residential Roof Edge Walkthrough",
            description="Inspect a short site report for obvious fall protection and ladder access violations on a roofing crew.",
        ),
        objective="Review the site report and submit the correct OSHA findings with citations, severity, evidence, and corrective actions.",
        inspector_role="You are the general contractor's safety inspector conducting a morning walkthrough.",
        max_steps=5,
        site_report=(
            "Morning report from Lot 14 townhouse project: two roofers were installing sheathing on a second-story roof edge "
            "about 18 feet above grade. The foreman wrote that no guardrails or personal fall arrest systems were in place "
            "because the crew expected to finish the edge work in less than fifteen minutes. Access to the roof was by an "
            "extension ladder that stopped roughly one foot below the landing point at the eave. One laborer held the ladder "
            "by hand while workers stepped onto the roof. Weather was dry and the roof deck was otherwise stable."
        ),
        references=REFERENCE_LIBRARY,
        target_findings=[
            TargetFinding(
                finding_id="roof_edge_no_fall_protection",
                label="Unprotected roof edge over 6 feet",
                label_aliases=["roof edge fall protection", "unguarded edge", "fall protection missing", "unprotected edge"],
                osha_citation="29 CFR 1926.501(b)(1)",
                citation_title="Unprotected sides and edges",
                severity="critical",
                evidence_keywords=["18 feet", "no guardrails", "no personal fall arrest", "roof edge"],
                corrective_keywords=["install guardrails", "personal fall arrest", "safety net", "protect workers from falls"],
            ),
            TargetFinding(
                finding_id="ladder_not_extending_3ft",
                label="Access ladder does not extend 3 feet above landing",
                label_aliases=["ladder extension too short", "ladder side rails too short", "ladder not 3 feet above landing"],
                osha_citation="29 CFR 1926.1053(b)(1)",
                citation_title="Portable ladder side rails",
                severity="high",
                evidence_keywords=["one foot below", "stopped roughly one foot below the landing", "extension ladder"],
                corrective_keywords=["extend ladder at least 3 feet", "secure grasping device", "proper ladder access"],
            ),
        ],
        grading_notes="Easy task with two high-salience hazards and direct evidence in the report.",
    ),
    "medium_trench_excavation_control": InspectionTask(
        descriptor=TaskDescriptor(
            name="medium_trench_excavation_control",
            difficulty="medium",
            title="Utility Trench Safety Check",
            description="Inspect a trenching report for cave-in, egress, and spoil pile violations.",
        ),
        objective="Identify all trenching hazards and issue the correct excavation citations with practical corrections.",
        inspector_role="You are the competent person reviewing a utility installation trench before the afternoon shift.",
        max_steps=6,
        site_report=(
            "Utility crew report for Maple Street sewer tie-in: employees were working inside a straight trench measured at "
            "approximately 7 feet deep and 60 feet long. The trench walls were cut nearly vertical in previously disturbed soil. "
            "The superintendent noted there was no trench box or shoring in place because the pipe fit-up was 'almost done.' "
            "Excavated spoil was piled right along the lip of the trench on the east side. The only ladder was set at the north "
            "end, and workers at the south end had to travel well over 25 feet to reach it. No standing water was present."
        ),
        references=REFERENCE_LIBRARY,
        target_findings=[
            TargetFinding(
                finding_id="trench_no_protective_system",
                label="Excavation lacks cave-in protective system",
                label_aliases=["trench cave-in protection missing", "no trench box", "no shoring", "unprotected excavation"],
                osha_citation="29 CFR 1926.652(a)(1)",
                citation_title="Protective systems in excavations",
                severity="critical",
                evidence_keywords=["7 feet deep", "no trench box", "no shoring", "previously disturbed soil", "vertical"],
                corrective_keywords=["install trench box", "use shoring", "slope or shore", "protect workers from cave-ins"],
            ),
            TargetFinding(
                finding_id="trench_egress_distance",
                label="Trench lacks safe egress within 25 feet",
                label_aliases=["egress too far", "ladder too far from workers", "no safe exit within 25 feet"],
                osha_citation="29 CFR 1926.651(c)(2)",
                citation_title="Means of egress in trenches",
                severity="high",
                evidence_keywords=["only ladder", "north end", "south end", "travel the full length"],
                corrective_keywords=["provide ladder", "safe egress within 25 feet", "additional access point"],
            ),
            TargetFinding(
                finding_id="spoil_pile_at_edge",
                label="Spoil pile stored at trench edge",
                label_aliases=["spoil pile too close", "material at excavation edge", "spoil on lip of trench"],
                osha_citation="29 CFR 1926.651(j)(2)",
                citation_title="Spoil piles kept back from edge",
                severity="medium",
                evidence_keywords=["spoil", "piled right along the lip", "trench edge"],
                corrective_keywords=["move spoil at least 2 feet back", "use retaining device", "keep material from falling in"],
            ),
        ],
        grading_notes="Medium task because the agent must identify three separate excavation rules with overlapping evidence.",
    ),
    "hard_scaffold_multi_hazard": InspectionTask(
        descriptor=TaskDescriptor(
            name="hard_scaffold_multi_hazard",
            difficulty="hard",
            title="Facade Scaffold and Interior Deck Inspection",
            description="Inspect a dense report that mixes scaffold, floor opening, and falling-object hazards.",
        ),
        objective="Separate multiple simultaneous fall hazards and assign the correct OSHA citations without over-citing.",
        inspector_role="You are an owner-side safety consultant auditing a mixed-use building renovation.",
        max_steps=7,
        site_report=(
            "Afternoon audit at the Jefferson mixed-use renovation: masonry workers were laying block from a supported scaffold "
            "set approximately 16 feet above the sidewalk. The working platform had open ends and no visible personal fall arrest "
            "equipment in use. On the level above, another crew stacked brick and buckets directly on a landing over the scaffold, "
            "and no canopy, toeboards, or barricaded drop zone had been installed to protect the masons below. Inside the building, "
            "drywall crews had removed a temporary plywood cover from a mechanical opening in the third-floor deck and left the "
            "opening uncovered while material was being staged nearby. Access to the scaffold itself was by an integrated frame "
            "ladder that crews used normally, and the weather was clear."
        ),
        references=REFERENCE_LIBRARY,
        target_findings=[
            TargetFinding(
                finding_id="scaffold_fall_protection",
                label="Workers on scaffold over 10 feet lack fall protection",
                label_aliases=["scaffold fall protection missing", "unguarded scaffold", "no scaffold fall protection"],
                osha_citation="29 CFR 1926.451(g)(1)",
                citation_title="Fall protection on scaffolds",
                severity="critical",
                evidence_keywords=["supported scaffold", "16 feet", "open ends", "no visible personal fall arrest"],
                corrective_keywords=["guardrail", "personal fall arrest", "protect scaffold workers from falling"],
            ),
            TargetFinding(
                finding_id="scaffold_falling_objects",
                label="Scaffold workers exposed to falling objects from above",
                label_aliases=["falling object protection missing", "materials above scaffold", "no toeboards or canopy"],
                osha_citation="29 CFR 1926.451(h)(1)",
                citation_title="Falling object protection on scaffolds",
                severity="high",
                evidence_keywords=["stacked brick", "buckets", "over the scaffold", "no canopy", "no toeboards", "masons below"],
                corrective_keywords=["install toeboards", "canopy", "barricade drop zone", "protect workers below"],
            ),
            TargetFinding(
                finding_id="floor_hole_unprotected",
                label="Uncovered floor opening exposes workers to fall",
                label_aliases=["floor hole unprotected", "mechanical opening uncovered", "opening in deck left uncovered"],
                osha_citation="29 CFR 1926.501(b)(4)(i)",
                citation_title="Floor holes",
                severity="high",
                evidence_keywords=["temporary plywood cover", "removed", "mechanical opening", "left uncovered", "third-floor deck"],
                corrective_keywords=["cover the hole", "guardrail around opening", "personal fall arrest", "protect workers from falling through"],
            ),
        ],
        grading_notes="Hard task because it mixes scaffold fall protection, falling object control, and an interior floor-hole hazard in one narrative.",
    ),
}


def list_tasks() -> List[TaskDescriptor]:
    return [task.descriptor for task in TASKS.values()]


def get_task(task_name: str) -> InspectionTask:
    try:
        return TASKS[task_name]
    except KeyError as exc:
        valid = ", ".join(sorted(TASKS))
        raise KeyError(f"Unknown task '{task_name}'. Valid tasks: {valid}") from exc
