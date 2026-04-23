from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class OSHARule:
    code: str
    title: str
    trigger: str
    correction_keywords: List[str]


OSHA_RULES: Dict[str, OSHARule] = {
    "OSHA 1926.502": OSHARule(
        code="OSHA 1926.502",
        title="Fall protection systems",
        trigger="Workers above 6 feet must have fall protection.",
        correction_keywords=["fall", "guardrail", "harness", "arrest", "protect"],
    ),
    "OSHA 1926.652": OSHARule(
        code="OSHA 1926.652",
        title="Excavation protective systems",
        trigger="Excavations deeper than 5 feet require shoring, shielding, or sloping.",
        correction_keywords=["shore", "shoring", "trench box", "slope", "shield"],
    ),
    "OSHA 1926.550": OSHARule(
        code="OSHA 1926.550",
        title="Crane swing radius",
        trigger="No workers may stand in a crane swing radius during lifts.",
        correction_keywords=["swing", "radius", "barricade", "spotter", "clear"],
    ),
    "OSHA 1910.147": OSHARule(
        code="OSHA 1910.147",
        title="Lockout/tagout",
        trigger="Electrical work needs lockout/tagout.",
        correction_keywords=["lockout", "tagout", "de-energize", "isolate"],
    ),
    "OSHA 1926.100": OSHARule(
        code="OSHA 1926.100",
        title="Head protection",
        trigger="Hard hats are mandatory in active construction zones.",
        correction_keywords=["hard hat", "helmet", "head protection"],
    ),
    "OSHA 1926.451": OSHARule(
        code="OSHA 1926.451",
        title="Scaffold inspection",
        trigger="Scaffolding must be inspected before each shift.",
        correction_keywords=["scaffold", "inspect", "inspection", "tag"],
    ),
    "OSHA 1926.150": OSHARule(
        code="OSHA 1926.150",
        title="Fire protection",
        trigger="Welding requires an extinguisher within 100 feet.",
        correction_keywords=["fire", "extinguisher", "hot work", "welding"],
    ),
    "OSHA 1926.32": OSHARule(
        code="OSHA 1926.32",
        title="Permit authorization",
        trigger="No construction work may start without the required valid permit.",
        correction_keywords=["permit", "approval", "authorize", "valid"],
    ),
}


def valid_rule_codes() -> List[str]:
    return sorted(OSHA_RULES)


def grade_incident_report(violation_code: str, corrective_action: str | None) -> bool:
    rule = OSHA_RULES.get(violation_code)
    if rule is None or not corrective_action:
        return False
    text = corrective_action.lower()
    return any(keyword in text for keyword in rule.correction_keywords)


def rule_summaries(codes: Iterable[str]) -> List[str]:
    summaries = []
    for code in codes:
        rule = OSHA_RULES.get(code)
        if rule:
            summaries.append(f"{rule.code}: {rule.title} - {rule.trigger}")
    return summaries
