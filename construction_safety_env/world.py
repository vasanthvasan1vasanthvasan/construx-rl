from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from .agents import SUBCONTRACTOR_AGENTS
from .models import (
    CrewSnapshot,
    Difficulty,
    MaterialOrder,
    OSHAAlert,
    PermitSnapshot,
    QuoteSnapshot,
    TaskSnapshot,
    WeatherDay,
)
from .osha import grade_incident_report
from .scenarios import ScenarioSpec, TaskSpec, get_scenario


MATERIAL_COSTS = {
    "concrete": 900,
    "steel": 1200,
    "lumber": 450,
    "roofing": 650,
    "wire": 500,
    "pipe": 520,
    "insulation": 260,
    "drywall": 300,
    "paint": 180,
}
QUALITY_MULTIPLIERS = {"cheap": 0.82, "standard": 1.0, "premium": 1.28}
ACTION_COST = {
    "check_weather": 0,
    "check_inventory": 0,
    "check_permit_status": 0,
    "request_inspection": 150,
    "request_quote": 0,
    "negotiate": 0,
    "file_incident_report": 100,
    "request_permit": 600,
    "order_material": 0,
    "hold_crew": 0,
    "assign_crew": 0,
    "accept_quote": 0,
}
DAY_ADVANCING_ACTIONS = {"assign_crew", "hold_crew", "accept_quote"}


@dataclass
class RuntimeTask:
    spec: TaskSpec
    status: str = "blocked"
    remaining_days: int = 0


@dataclass
class RuntimeCrew:
    crew_id: str
    crew_type: str
    status: str
    assigned_task: Optional[str]
    daily_cost: int
    idle_cost: int


class ConstructionWorld:
    def __init__(self, difficulty: Difficulty, seed: int, memory_hint: str | None = None) -> None:
        self.rng = random.Random(seed)
        self.scenario: ScenarioSpec = get_scenario(difficulty)
        self.difficulty = difficulty
        self.day = 1
        self.step_index = 0
        self.budget = self.scenario.starting_budget
        self.tasks = {task.task_id: RuntimeTask(task, "blocked", task.duration) for task in self.scenario.tasks}
        self.crews = {
            crew.crew_id: RuntimeCrew(crew.crew_id, crew.crew_type, "available", None, crew.daily_cost, crew.idle_cost)
            for crew in self.scenario.crews
        }
        self.inventory = dict(self.scenario.initial_inventory)
        self.pending_orders: List[MaterialOrder] = []
        self.permits = {
            permit_type: PermitSnapshot(permit_id=f"{permit_type}-permit", permit_type=permit_type, status="not_requested")
            for permit_type in self.scenario.permit_types
        }
        self.alerts: List[OSHAAlert] = []
        self.quotes: Dict[str, QuoteSnapshot] = {}
        self.site_log: List[str] = []
        self.memory_hint = memory_hint
        self.last_weather_check_day: Optional[int] = None
        self.inspected_zones: Dict[str, int] = {}
        self.done = False
        self.success = False
        self.last_error: Optional[str] = None
        self._weather = self._make_weather(seed)
        self._refresh_task_statuses()

    @property
    def max_days(self) -> int:
        return self.scenario.max_days

    @property
    def max_steps(self) -> int:
        return self.scenario.max_steps

    def weather_forecast(self) -> List[WeatherDay]:
        return [self._weather[min(self.max_days, self.day + offset)] for offset in range(3)]

    def task_snapshots(self) -> Dict[str, TaskSnapshot]:
        self._refresh_task_statuses()
        return {task_id: self._snapshot_task(runtime) for task_id, runtime in self.tasks.items()}

    def crew_snapshots(self) -> Dict[str, CrewSnapshot]:
        return {
            crew_id: CrewSnapshot(
                crew_id=crew.crew_id,
                crew_type=crew.crew_type,
                status=crew.status,  # type: ignore[arg-type]
                assigned_task=crew.assigned_task,
                daily_cost=crew.daily_cost,
                idle_cost=crew.idle_cost,
            )
            for crew_id, crew in self.crews.items()
        }

    def completed_tasks(self) -> List[str]:
        return [task_id for task_id, task in self.tasks.items() if task.status == "done"]

    def failed_tasks(self) -> List[str]:
        return [task_id for task_id, task in self.tasks.items() if task.status == "failed"]

    def safety_violations(self) -> int:
        return len(self.alerts)

    def missing_reports(self) -> int:
        return sum(1 for alert in self.alerts if not alert.incident_report_filed)

    def apply(self, action_type: str, payload: dict) -> dict:
        self.last_error = None
        self.step_index += 1
        self._settle_start_of_day()
        before_completed = len(self.completed_tasks())

        handler = getattr(self, f"_act_{action_type}", None)
        if handler is None:
            self.last_error = f"Unsupported action_type: {action_type}"
            info = {"error": self.last_error}
        else:
            try:
                info = handler(**payload)
            except ValueError as exc:
                self.last_error = str(exc)
                self.site_log.append(f"Day {self.day}: blocked action - {self.last_error}")
                info = {"error": self.last_error}

        self.budget -= ACTION_COST.get(action_type, 0)
        self._refresh_task_statuses()
        immediate_success = all(task.status == "done" for task in self.tasks.values()) and self.budget > 0
        if action_type in DAY_ADVANCING_ACTIONS and not immediate_success:
            self._advance_day()
        else:
            self._settle_start_of_day()
        self._refresh_task_statuses()
        self.success = all(task.status == "done" for task in self.tasks.values()) and self.budget > 0
        self.done = self.success or self.budget <= 0 or self.day > self.max_days or self.step_index >= self.max_steps
        if self.budget <= 0:
            self.site_log.append(f"Day {self.day}: project failed because budget reached zero.")

        info["new_tasks_completed"] = len(self.completed_tasks()) - before_completed
        info["success"] = self.success
        return info

    def _act_check_weather(self, **_: object) -> dict:
        self.last_weather_check_day = self.day
        forecast = [weather.model_dump() for weather in self.weather_forecast()]
        self.site_log.append(f"Day {self.day}: weather checked for days {self.day}-{min(self.max_days, self.day + 2)}.")
        return {"forecast": forecast}

    def _act_check_inventory(self, **_: object) -> dict:
        self.site_log.append(f"Day {self.day}: inventory checked.")
        return {"inventory": dict(self.inventory), "pending_orders": [order.model_dump() for order in self.pending_orders]}

    def _act_request_permit(self, permit_type: Optional[str] = None, **_: object) -> dict:
        if permit_type not in self.permits:
            raise ValueError(f"Unknown permit_type: {permit_type}")
        permit = self.permits[permit_type]
        if permit.status == "approved":
            return {"permit": permit.model_dump(), "note": "already approved"}
        permit.status = "pending"
        permit.requested_day = self.day
        permit.approval_day = self.day + (2 if self.difficulty == "easy" else 3)
        self.site_log.append(f"Day {self.day}: requested {permit_type} permit; expected approval day {permit.approval_day}.")
        return {"permit": permit.model_dump()}

    def _act_check_permit_status(self, permit_id: Optional[str] = None, permit_type: Optional[str] = None, **_: object) -> dict:
        permit = self._find_permit(permit_id, permit_type)
        self.site_log.append(f"Day {self.day}: checked {permit.permit_type} permit status: {permit.status}.")
        return {"permit": permit.model_dump()}

    def _act_order_material(
        self,
        material: Optional[str] = None,
        quantity: Optional[int] = None,
        quality: Optional[str] = None,
        **_: object,
    ) -> dict:
        if material not in MATERIAL_COSTS or not quantity or not quality:
            raise ValueError("order_material requires material, quantity, and quality")
        cost = int(MATERIAL_COSTS[material] * quantity * QUALITY_MULTIPLIERS[quality])
        if cost > self.budget:
            raise ValueError(f"Insufficient budget for {material} order costing {cost}")
        self.budget -= cost
        eta = self.day + 3
        order = MaterialOrder(material=material, quantity=quantity, quality=quality, eta_day=eta, cost=cost)  # type: ignore[arg-type]
        self.pending_orders.append(order)
        self.site_log.append(f"Day {self.day}: ordered {quantity} {quality} {material}; ETA day {eta}.")
        return {"order": order.model_dump()}

    def _act_hold_crew(self, crew_id: Optional[str] = None, reason: Optional[str] = None, **_: object) -> dict:
        crew = self._require_crew(crew_id)
        crew.status = "held"
        crew.assigned_task = None
        self.site_log.append(f"Day {self.day}: held {crew.crew_id} crew. Reason: {reason or 'not specified'}.")
        return {"crew": crew.crew_id, "status": "held"}

    def _act_assign_crew(self, crew_id: Optional[str] = None, task_id: Optional[str] = None, **_: object) -> dict:
        crew = self._require_crew(crew_id)
        task = self._require_task(task_id)
        if task.spec.required_crew == "subcontractor":
            raise ValueError("Task requires an accepted subcontractor quote, not a direct crew assignment")
        if crew.crew_type != task.spec.required_crew:
            raise ValueError(f"{crew.crew_id} cannot perform {task.spec.required_crew} task {task.spec.task_id}")
        starting_fresh = task.status == "available"
        self._validate_task_start(task, consume_materials=starting_fresh)
        crew.status = "assigned"
        crew.assigned_task = task.spec.task_id
        task.status = "in_progress"
        task.remaining_days -= 1
        self.budget -= crew.daily_cost
        if task.remaining_days <= 0:
            task.status = "done"
            crew.status = "available"
            crew.assigned_task = None
            self.site_log.append(f"Day {self.day}: {crew.crew_id} completed {task.spec.task_id}.")
        else:
            self.site_log.append(f"Day {self.day}: {crew.crew_id} progressed {task.spec.task_id}; {task.remaining_days} day(s) remain.")
        return {"task_id": task.spec.task_id, "task_status": task.status}

    def _act_request_inspection(self, zone: Optional[str] = None, **_: object) -> dict:
        if not zone:
            raise ValueError("request_inspection requires zone")
        self.inspected_zones[zone] = self.day
        self.site_log.append(f"Day {self.day}: inspection requested for {zone}.")
        return {"zone": zone, "inspection_day": self.day}

    def _act_file_incident_report(
        self,
        violation_code: Optional[str] = None,
        crew_id: Optional[str] = None,
        corrective_action: Optional[str] = None,
        **_: object,
    ) -> dict:
        if not violation_code:
            raise ValueError("file_incident_report requires violation_code")
        matching = [alert for alert in self.alerts if alert.violation_code == violation_code and not alert.incident_report_filed]
        if not matching:
            raise ValueError(f"No open incident for {violation_code}")
        accepted = grade_incident_report(violation_code, corrective_action)
        if accepted:
            matching[0].incident_report_filed = True
        self.site_log.append(f"Day {self.day}: incident report for {violation_code} {'accepted' if accepted else 'rejected'}.")
        return {"accepted": accepted, "crew_id": crew_id}

    def _act_request_quote(self, subcontractor_id: Optional[str] = None, task_id: Optional[str] = None, **_: object) -> dict:
        if subcontractor_id not in SUBCONTRACTOR_AGENTS:
            raise ValueError(f"Unknown subcontractor_id: {subcontractor_id}")
        task = self._require_task(task_id)
        sub = SUBCONTRACTOR_AGENTS[subcontractor_id]
        if task.spec.task_id != sub.specialty:
            raise ValueError(f"{subcontractor_id} does not cover {task.spec.task_id}")
        base = 5200 + 1400 * task.spec.duration
        quote_id = f"{subcontractor_id}-{task.spec.task_id}-{self.day}"
        quote = QuoteSnapshot(
            quote_id=quote_id,
            subcontractor_id=subcontractor_id,
            task_id=task.spec.task_id,
            price=sub.quote_price(base, self.day),
            available_day=self.day + sub.availability_offset,
            status="open",
        )
        self.quotes[quote_id] = quote
        self.site_log.append(f"Day {self.day}: quote {quote_id} opened at {quote.price}.")
        return {"quote": quote.model_dump()}

    def _act_negotiate(self, quote_id: Optional[str] = None, counter_price: Optional[int] = None, **_: object) -> dict:
        quote = self._require_quote(quote_id)
        if not counter_price:
            raise ValueError("negotiate requires counter_price")
        sub = SUBCONTRACTOR_AGENTS[quote.subcontractor_id]
        accepted, new_price = sub.counter(counter_price, quote.price)
        quote.price = new_price
        if not accepted:
            self.site_log.append(f"Day {self.day}: {quote.subcontractor_id} countered quote {quote.quote_id} at {new_price}.")
        else:
            self.site_log.append(f"Day {self.day}: {quote.subcontractor_id} accepted negotiated price {new_price}.")
        return {"accepted": accepted, "quote": quote.model_dump()}

    def _act_accept_quote(self, quote_id: Optional[str] = None, **_: object) -> dict:
        quote = self._require_quote(quote_id)
        if quote.price > self.budget:
            raise ValueError(f"Insufficient budget to accept quote {quote.quote_id}")
        task = self._require_task(quote.task_id)
        if self.day < quote.available_day:
            raise ValueError(f"Subcontractor not available until day {quote.available_day}")
        self._validate_task_start(task, consume_materials=True)
        self.budget -= quote.price
        quote.status = "accepted"
        task.status = "done"
        task.remaining_days = 0
        self.site_log.append(f"Day {self.day}: accepted quote {quote.quote_id}; {task.spec.task_id} completed.")
        return {"quote": quote.model_dump(), "task_status": "done"}

    def _advance_day(self) -> None:
        idle_cost = 0
        for crew in self.crews.values():
            if crew.status in {"available", "held"}:
                idle_cost += crew.idle_cost
            if crew.status == "held":
                crew.status = "available"
        self.budget -= idle_cost
        self._settle_start_of_day()
        self.day += 1

    def _settle_start_of_day(self) -> None:
        for order in list(self.pending_orders):
            if order.eta_day <= self.day:
                self.inventory[order.material] = self.inventory.get(order.material, 0) + order.quantity
                self.pending_orders.remove(order)
                self.site_log.append(f"Day {self.day}: received {order.quantity} {order.material}.")

        for permit in self.permits.values():
            if permit.status == "pending" and permit.approval_day is not None and self.day >= permit.approval_day:
                permit.status = "approved"
                self.site_log.append(f"Day {self.day}: {permit.permit_type} permit approved.")

    def _validate_task_start(self, task: RuntimeTask, consume_materials: bool) -> None:
        self._refresh_task_statuses()
        if task.status not in {"available", "in_progress"}:
            raise ValueError(f"Task {task.spec.task_id} is blocked: {', '.join(self._blocked_reasons(task))}")
        weather = self._weather[self.day]
        if task.spec.outdoor and weather.rain_probability >= 0.65:
            self._add_violation("OSHA 1926.100", task, "Outdoor work attempted during unsafe storm conditions.")
            raise ValueError(f"Cannot perform outdoor work in rain risk {weather.rain_probability:.2f}")
        if task.spec.task_id == "foundation_pour" and weather.rain_probability > 0.35:
            raise ValueError("Cannot pour concrete in rain.")
        if task.spec.crane and weather.wind_mph > 25:
            self._add_violation("OSHA 1926.550", task, "Crane operation attempted above 25 mph wind.")
            raise ValueError(f"Cannot operate crane at {weather.wind_mph} mph wind.")
        for code in task.spec.osha_rules:
            if code == "OSHA 1926.502" and task.spec.height_ft > 6 and task.spec.zone not in self.inspected_zones:
                self._add_violation(code, task, "Work above 6 feet started without fall-protection inspection.")
            if code == "OSHA 1926.652" and task.spec.excavation_depth_ft > 5 and task.spec.zone not in self.inspected_zones:
                self._add_violation(code, task, "Deep excavation started without shoring inspection.")
            if code == "OSHA 1910.147" and task.spec.electrical and task.spec.zone not in self.inspected_zones:
                self._add_violation(code, task, "Electrical work started without lockout/tagout inspection.")
            if code == "OSHA 1926.451" and task.spec.scaffold and task.spec.zone not in self.inspected_zones:
                self._add_violation(code, task, "Scaffold work started without inspection.")
            if code == "OSHA 1926.150" and task.spec.welding and task.spec.zone not in self.inspected_zones:
                self._add_violation(code, task, "Welding started without fire protection inspection.")

        if consume_materials:
            for material, qty in task.spec.required_materials.items():
                self.inventory[material] = self.inventory.get(material, 0) - qty

    def _blocked_reasons(self, task: RuntimeTask) -> List[str]:
        reasons = []
        for prereq in task.spec.prerequisites:
            if prereq in self.tasks and self.tasks[prereq].status != "done":
                reasons.append(f"needs {prereq}")
        if task.spec.required_permit:
            permit = self.permits.get(task.spec.required_permit)
            if permit is None or permit.status != "approved":
                reasons.append(f"needs approved {task.spec.required_permit} permit")
        for material, qty in task.spec.required_materials.items():
            if self.inventory.get(material, 0) < qty:
                reasons.append(f"needs {qty} {material}")
        return reasons

    def _refresh_task_statuses(self) -> None:
        for task in self.tasks.values():
            if task.status in {"done", "failed", "in_progress"}:
                continue
            task.status = "available" if not self._blocked_reasons(task) else "blocked"

    def _snapshot_task(self, task: RuntimeTask) -> TaskSnapshot:
        return TaskSnapshot(
            task_id=task.spec.task_id,
            name=task.spec.name,
            level=task.spec.level,
            zone=task.spec.zone,
            status=task.status,  # type: ignore[arg-type]
            prerequisites=task.spec.prerequisites,
            remaining_days=task.remaining_days,
            required_crew=task.spec.required_crew,
            required_materials=task.spec.required_materials,
            required_permit=task.spec.required_permit,
            osha_rules=task.spec.osha_rules,
            blocked_reasons=self._blocked_reasons(task),
        )

    def _add_violation(self, code: str, task: RuntimeTask, description: str) -> None:
        alert = OSHAAlert(violation_code=code, crew_id=None, task_id=task.spec.task_id, description=description)
        self.alerts.append(alert)
        self.site_log.append(f"Day {self.day}: violation {code} on {task.spec.task_id}: {description}")

    def _require_crew(self, crew_id: Optional[str]) -> RuntimeCrew:
        if crew_id not in self.crews:
            raise ValueError(f"Unknown crew_id: {crew_id}")
        return self.crews[crew_id]

    def _require_task(self, task_id: Optional[str]) -> RuntimeTask:
        if task_id not in self.tasks:
            raise ValueError(f"Unknown task_id: {task_id}")
        return self.tasks[task_id]

    def _require_quote(self, quote_id: Optional[str]) -> QuoteSnapshot:
        if quote_id not in self.quotes:
            raise ValueError(f"Unknown quote_id: {quote_id}")
        return self.quotes[quote_id]

    def _find_permit(self, permit_id: Optional[str], permit_type: Optional[str]) -> PermitSnapshot:
        if permit_type in self.permits:
            return self.permits[permit_type]
        for permit in self.permits.values():
            if permit.permit_id == permit_id:
                return permit
        raise ValueError(f"Unknown permit: {permit_id or permit_type}")

    def _make_weather(self, seed: int) -> Dict[int, WeatherDay]:
        rng = random.Random(seed + 17)
        weather: Dict[int, WeatherDay] = {}
        for day in range(1, self.scenario.max_days + 3):
            if self.difficulty == "easy":
                rain = 0.05
                wind = 8 + day % 4
            else:
                rain = round(rng.choice([0.05, 0.10, 0.20, 0.35, 0.70 if day in {8, 14, 21} else 0.25]), 2)
                wind = rng.choice([8, 12, 16, 22, 29 if day in {10, 17, 24} else 18])
            summary = "storm" if rain >= 0.65 else "wind hold" if wind > 25 else "clear"
            weather[day] = WeatherDay(day=day, rain_probability=rain, wind_mph=wind, summary=summary)
        return weather
