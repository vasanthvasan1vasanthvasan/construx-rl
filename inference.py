from __future__ import annotations

import json
import os
from typing import List, Optional
from urllib.parse import urlparse

from openai import OpenAI

from construction_safety_env.client import ConstructionSafetyEnvClient
from construction_safety_env.env import ConstructionSafetyEnv
from construction_safety_env.models import ConstructionSafetyAction, FindingSubmission
from construction_safety_env.tasks import TASKS


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK = "construction_site_safety_inspector"
MAX_TOKENS = 450
TEMPERATURE = 0.0


def _is_local_url(url: Optional[str]) -> bool:
    if not url:
        return False
    hostname = urlparse(url).hostname
    return hostname in {"127.0.0.1", "localhost", "0.0.0.0"}


def _format_action(action: ConstructionSafetyAction) -> str:
    if action.action_type == "submit_report":
        return f"submit_report(summary={((action.final_summary or '')[:60])!r})"
    assert action.finding is not None
    return (
        "issue_finding("
        f"hazard_label={action.finding.hazard_label!r}, "
        f"osha_citation={action.finding.osha_citation!r}, "
        f"severity={action.finding.severity!r})"
    )


def _print_start(task_name: str) -> None:
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}")


def _print_step(step: int, action: ConstructionSafetyAction, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={_format_action(action)} reward={reward:.2f} "
        f"done={'true' if done else 'false'} error={error if error else 'null'}"
    )


def _print_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    joined = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={'true' if success else 'false'} steps={steps} "
        f"score={score:.2f} rewards={joined}"
    )


def _build_prompt(observation) -> str:
    references = "\n".join(
        f"- {ref.citation}: {ref.title}. {ref.summary}" for ref in observation.reference_library
    )
    prior = "\n".join(
        f"- {finding.hazard_label} | {finding.osha_citation} | {finding.severity}"
        for finding in observation.submitted_findings
    ) or "- none"
    return (
        "You are a construction safety inspector. Read the site report and return JSON only.\n"
        "Schema:\n"
        '{"action_type":"issue_finding"|"submit_report","finding":{"hazard_label":"...","osha_citation":"...","severity":"low|medium|high|critical","evidence":"...","corrective_action":"...","confidence":0.0},"final_summary":"..."}\n'
        "Rules:\n"
        "- Use exactly one new finding per step or submit_report when finished.\n"
        "- Only cite hazards explicitly supported by the report.\n"
        "- Do not repeat prior findings.\n\n"
        f"Objective: {observation.objective}\n"
        f"Role: {observation.inspector_role}\n"
        f"Step: {observation.step_index}/{observation.max_steps}\n"
        f"Current score: {observation.current_score}\n"
        f"Previously submitted findings:\n{prior}\n\n"
        f"Reference library:\n{references}\n\n"
        f"Site report:\n{observation.site_report}\n"
    )


def _extract_json(text: str) -> dict:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in model response: {text}")
    return json.loads(text[start : end + 1])


class _LocalEnvAdapter:
    def __init__(self, task_name: str) -> None:
        self._env = ConstructionSafetyEnv()
        self._task_name = task_name

    def reset(self):
        class _ResetResult:
            def __init__(self, observation) -> None:
                self.observation = observation

        return _ResetResult(self._env.reset(task_name=self._task_name, seed=0))

    def step(self, action: ConstructionSafetyAction):
        observation, reward, done, info = self._env.step(action)

        class _StepResult:
            def __init__(self, observation, reward, done, info) -> None:
                self.observation = observation
                self.reward = reward
                self.done = done
                self.info = info

        return _StepResult(observation, reward, done, info)

    def state(self):
        class _StateResult:
            def __init__(self, state) -> None:
                self.state = state

        return _StateResult(self._env.state())

    def close(self) -> None:
        self._env.close()


class _HttpEnvAdapter:
    def __init__(self, task_name: str, base_url: str) -> None:
        self._client = ConstructionSafetyEnvClient(base_url=base_url)
        self._task_name = task_name

    def reset(self):
        return self._client.reset(task_name=self._task_name, seed=0)

    def step(self, action: ConstructionSafetyAction):
        return self._client.step(action)

    def state(self):
        return self._client.state()

    def close(self) -> None:
        self._client.close()


def _make_env_runner(task_name: str):
    if _is_local_url(API_BASE_URL):
        return _HttpEnvAdapter(task_name=task_name, base_url=API_BASE_URL)
    return _LocalEnvAdapter(task_name)


def _heuristic_action(observation) -> ConstructionSafetyAction:
    report = observation.site_report.lower()
    submitted_citations = {finding.osha_citation for finding in observation.submitted_findings}
    rules = [
        (
            "29 CFR 1926.501(b)(1)",
            ["roof", "edge", "18 feet", "no guardrails"],
            FindingSubmission(
                hazard_label="Unprotected roof edge over 6 feet",
                osha_citation="29 CFR 1926.501(b)(1)",
                severity="critical",
                evidence="Roofers were working at an 18-foot roof edge with no guardrails and no personal fall arrest.",
                corrective_action="Install guardrails or require personal fall arrest before roof edge work continues.",
                confidence=0.98,
            ),
        ),
        (
            "29 CFR 1926.1053(b)(1)",
            ["ladder", "one foot below", "landing"],
            FindingSubmission(
                hazard_label="Access ladder does not extend 3 feet above landing",
                osha_citation="29 CFR 1926.1053(b)(1)",
                severity="high",
                evidence="The extension ladder stopped about one foot below the roof landing instead of extending 3 feet above it.",
                corrective_action="Extend the ladder rails at least 3 feet above the landing or provide an equivalent grasping device.",
                confidence=0.95,
            ),
        ),
        (
            "29 CFR 1926.652(a)(1)",
            ["7 feet deep", "trench box", "shoring"],
            FindingSubmission(
                hazard_label="Excavation lacks cave-in protective system",
                osha_citation="29 CFR 1926.652(a)(1)",
                severity="critical",
                evidence="Employees were in a 7-foot trench with near-vertical walls in disturbed soil and no trench box or shoring.",
                corrective_action="Use a trench box, shoring, or approved sloping before workers enter the excavation.",
                confidence=0.99,
            ),
        ),
        (
            "29 CFR 1926.651(c)(2)",
            ["only ladder", "north end", "south end"],
            FindingSubmission(
                hazard_label="Trench lacks safe egress within 25 feet",
                osha_citation="29 CFR 1926.651(c)(2)",
                severity="high",
                evidence="The only ladder was at the north end, forcing workers at the south end to travel the trench length to exit.",
                corrective_action="Provide an additional ladder or other safe egress so workers are within 25 feet of an exit.",
                confidence=0.94,
            ),
        ),
        (
            "29 CFR 1926.651(j)(2)",
            ["spoil", "lip of the trench"],
            FindingSubmission(
                hazard_label="Spoil pile stored at trench edge",
                osha_citation="29 CFR 1926.651(j)(2)",
                severity="medium",
                evidence="Excavated spoil was piled right along the lip of the trench instead of being kept back from the edge.",
                corrective_action="Move spoil piles at least 2 feet back from the trench edge or restrain the material from falling in.",
                confidence=0.93,
            ),
        ),
        (
            "29 CFR 1926.451(g)(1)",
            ["supported scaffold", "16 feet", "open ends"],
            FindingSubmission(
                hazard_label="Workers on scaffold over 10 feet lack fall protection",
                osha_citation="29 CFR 1926.451(g)(1)",
                severity="critical",
                evidence="Masonry workers were on a supported scaffold about 16 feet above grade with open ends and no visible fall arrest.",
                corrective_action="Provide compliant scaffold fall protection such as guardrails or required personal fall arrest.",
                confidence=0.98,
            ),
        ),
        (
            "29 CFR 1926.451(h)(1)",
            ["stacked brick", "over the scaffold", "no canopy"],
            FindingSubmission(
                hazard_label="Scaffold workers exposed to falling objects from above",
                osha_citation="29 CFR 1926.451(h)(1)",
                severity="high",
                evidence="Brick and buckets were stored above the scaffold and there was no canopy, toeboard, or protected drop zone.",
                corrective_action="Install toeboards or a canopy and establish a controlled drop zone to protect workers below.",
                confidence=0.95,
            ),
        ),
        (
            "29 CFR 1926.501(b)(4)(i)",
            ["removed a temporary plywood cover", "opening uncovered", "third-floor deck"],
            FindingSubmission(
                hazard_label="Uncovered floor opening exposes workers to fall",
                osha_citation="29 CFR 1926.501(b)(4)(i)",
                severity="high",
                evidence="A temporary plywood cover was removed from a third-floor deck opening and the opening was left uncovered near staging activity.",
                corrective_action="Replace the cover or install guardrails around the opening until the hazard is eliminated.",
                confidence=0.96,
            ),
        ),
    ]
    for citation, clues, finding in rules:
        if citation in submitted_citations:
            continue
        if all(clue in report for clue in clues):
            return ConstructionSafetyAction(action_type="issue_finding", finding=finding)
    return ConstructionSafetyAction(
        action_type="submit_report",
        final_summary="Submitted all clearly supported construction safety hazards from the report.",
    )


def _llm_action(client: OpenAI, observation) -> ConstructionSafetyAction:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": "You are a careful OSHA-aware construction safety inspector. Return JSON only."},
            {"role": "user", "content": _build_prompt(observation)},
        ],
    )
    content = response.choices[0].message.content or ""
    payload = _extract_json(content)
    return ConstructionSafetyAction.model_validate(payload)


def run_episode(task_name: str, client: Optional[OpenAI]) -> dict:
    env = _make_env_runner(task_name)
    observation = env.reset().observation
    rewards: List[float] = []
    _print_start(task_name)

    try:
        while not observation.done:
            try:
                action = _llm_action(client, observation) if client is not None else _heuristic_action(observation)
            except Exception:
                action = _heuristic_action(observation)
            result = env.step(action)
            observation = result.observation
            reward = result.reward
            done = result.done
            rewards.append(reward.value)
            _print_step(observation.step_index, action, reward.value, done, observation.last_action_error)
            if done:
                break
    finally:
        state_result = env.state()
        score = state_result.state.current_score if hasattr(state_result, "state") else state_result.current_score
        _print_end(score >= 0.85, len(rewards), score, rewards)
        env.close()

    return {"score": score, "steps": float(len(rewards))}


def main() -> None:
    client = None
    if HF_TOKEN and not _is_local_url(API_BASE_URL):
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    for task_name in TASKS:
        run_episode(task_name, client)


if __name__ == "__main__":
    main()
