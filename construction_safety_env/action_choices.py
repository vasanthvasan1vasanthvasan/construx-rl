from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .models import ConstruxAction


@dataclass(frozen=True)
class ActionChoice:
    choice_id: int
    label: str
    action: ConstruxAction


def _action_signature(action: ConstruxAction) -> tuple:
    payload = action.model_dump(exclude_none=True)
    action_type = payload.pop("action_type")
    return (action_type, tuple(sorted(payload.items())))


def format_action_payload(action: ConstruxAction) -> str:
    payload = action.model_dump(exclude_none=True)
    action_type = payload.pop("action_type")
    args = ", ".join(f"{key}={value!r}" for key, value in payload.items())
    return f"{action_type}({args})" if args else f"{action_type}()"


def _add_choice(choices: List[ActionChoice], seen: set, action: ConstruxAction, label: str) -> None:
    signature = _action_signature(action)
    if signature in seen:
        return
    seen.add(signature)
    choices.append(ActionChoice(choice_id=len(choices) + 1, label=label, action=action))


def build_action_choices(observation, max_choices: int = 16) -> List[ActionChoice]:
    choices: List[ActionChoice] = []
    seen = set()

    _add_choice(choices, seen, ConstruxAction(action_type="check_weather"), "Check the 3-day weather forecast.")
    _add_choice(choices, seen, ConstruxAction(action_type="check_inventory"), "Check inventory and pending orders.")

    for permit_type, permit in observation.permits.items():
        if permit.status == "not_requested":
            _add_choice(
                choices,
                seen,
                ConstruxAction(action_type="request_permit", permit_type=permit_type),
                f"Request the {permit_type} permit.",
            )
        elif permit.status == "pending":
            _add_choice(
                choices,
                seen,
                ConstruxAction(action_type="check_permit_status", permit_type=permit_type),
                f"Check status of the {permit_type} permit.",
            )

    needed_materials = []
    for task in observation.tasks.values():
        if task.status in {"done", "failed"}:
            continue
        for material, qty in task.required_materials.items():
            pending = sum(order.quantity for order in observation.pending_orders if order.material == material)
            have = observation.inventory.get(material, 0) + pending
            if have < qty:
                needed_materials.append((task.task_id, material, qty - have))
    for task_id, material, shortage in needed_materials[:3]:
        _add_choice(
            choices,
            seen,
            ConstruxAction(action_type="order_material", material=material, quantity=max(1, shortage), quality="standard"),
            f"Order {max(1, shortage)} units of {material} for task {task_id}.",
        )

    for alert in observation.osha_alerts:
        if not alert.incident_report_filed:
            _add_choice(
                choices,
                seen,
                ConstruxAction(
                    action_type="file_incident_report",
                    violation_code=alert.violation_code,
                    crew_id=alert.crew_id,
                    corrective_action="Correct the cited hazard and retrain the crew before work resumes.",
                ),
                f"File incident report for {alert.violation_code}.",
            )

    inspection_candidates = []
    for task in observation.tasks.values():
        if task.status == "available" and task.osha_rules and task.zone not in {"office", "site"} and task.zone not in observation.inspected_zones:
            inspection_candidates.append((task.zone, task.task_id))
    for zone, task_id in inspection_candidates[:3]:
        _add_choice(
            choices,
            seen,
            ConstruxAction(action_type="request_inspection", zone=zone),
            f"Request inspection for zone {zone} before task {task_id}.",
        )

    for task in observation.tasks.values():
        if task.status == "available" and task.required_crew == "subcontractor":
            matching_quotes = [quote for quote in observation.subcontractor_quotes.values() if quote.task_id == task.task_id and quote.status == "open"]
            if matching_quotes:
                quote = sorted(matching_quotes, key=lambda item: item.price)[0]
                _add_choice(
                    choices,
                    seen,
                    ConstruxAction(action_type="accept_quote", quote_id=quote.quote_id),
                    f"Accept subcontractor quote {quote.quote_id} for task {task.task_id}.",
                )
            else:
                subcontractor_id = "weldco" if "welding" in task.task_id else "rapid_roof"
                _add_choice(
                    choices,
                    seen,
                    ConstruxAction(action_type="request_quote", subcontractor_id=subcontractor_id, task_id=task.task_id),
                    f"Request a quote from {subcontractor_id} for task {task.task_id}.",
                )

    for crew in observation.crews.values():
        if crew.status == "assigned" and crew.assigned_task:
            _add_choice(
                choices,
                seen,
                ConstruxAction(action_type="assign_crew", crew_id=crew.crew_id, task_id=crew.assigned_task),
                f"Continue {crew.crew_id} on task {crew.assigned_task}.",
            )

    available_tasks = [task for task in observation.tasks.values() if task.status == "available"]
    for crew in observation.crews.values():
        for task in available_tasks:
            if crew.crew_type == task.required_crew:
                _add_choice(
                    choices,
                    seen,
                    ConstruxAction(action_type="assign_crew", crew_id=crew.crew_id, task_id=task.task_id),
                    f"Assign {crew.crew_id} to task {task.task_id}.",
                )
                break

    for crew in observation.crews.values():
        if crew.status in {"available", "held"}:
            _add_choice(
                choices,
                seen,
                ConstruxAction(
                    action_type="hold_crew",
                    crew_id=crew.crew_id,
                    reason="Waiting for dependency, permit, material, weather, or subcontractor availability.",
                ),
                f"Hold crew {crew.crew_id} and wait for dependencies or safer conditions.",
            )

    return choices[:max_choices]


def format_action_choices(choices: List[ActionChoice]) -> str:
    return "\n".join(
        f"{choice.choice_id}. {format_action_payload(choice.action)} -- {choice.label}"
        for choice in choices
    )


def select_choice_for_action(choices: List[ActionChoice], action: ConstruxAction) -> ActionChoice:
    signature = _action_signature(action)
    for choice in choices:
        if _action_signature(choice.action) == signature:
            return choice
    fallback = ActionChoice(choice_id=len(choices) + 1, label="Fallback heuristic action.", action=action)
    return fallback


def parse_choice_response(text: str, choices: List[ActionChoice]) -> ConstruxAction:
    digits = "".join(ch for ch in str(text) if ch.isdigit())
    if not digits:
        raise ValueError(f"No numeric choice found in response: {text!r}")
    choice_id = int(digits)
    for choice in choices:
        if choice.choice_id == choice_id:
            return choice.action
    raise ValueError(f"Choice {choice_id} is not in the offered candidate list.")
