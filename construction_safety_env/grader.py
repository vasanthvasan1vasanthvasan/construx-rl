from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import re

from .models import FindingSubmission
from .tasks import InspectionTask, TargetFinding


TOKEN_RE = re.compile(r"[a-z0-9]+")
MIN_TASK_SCORE = 0.01
MAX_TASK_SCORE = 0.99


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _tokens(text: str) -> List[str]:
    return TOKEN_RE.findall(_normalize(text))


def _keyword_hit_ratio(text: str, keywords: Sequence[str]) -> float:
    normalized = _normalize(text)
    token_set = set(_tokens(normalized))
    hits = 0
    for keyword in keywords:
        keyword_norm = _normalize(keyword)
        if keyword_norm in normalized:
            hits += 1
            continue
        keyword_tokens = set(_tokens(keyword_norm))
        if keyword_tokens and keyword_tokens.issubset(token_set):
            hits += 1
    if not keywords:
        return 0.0
    return hits / len(keywords)


def _label_score(submission: FindingSubmission, target: TargetFinding) -> float:
    label = _normalize(submission.hazard_label)
    aliases = [target.label] + target.label_aliases
    for alias in aliases:
        alias_norm = _normalize(alias)
        if alias_norm in label or label in alias_norm:
            return 1.0
        alias_tokens = set(_tokens(alias_norm))
        label_tokens = set(_tokens(label))
        if alias_tokens and label_tokens:
            overlap = len(alias_tokens & label_tokens) / len(alias_tokens | label_tokens)
            if overlap >= 0.45:
                return 0.85
    return 0.0


def score_submission_against_target(submission: FindingSubmission, target: TargetFinding) -> float:
    label_component = 0.30 * _label_score(submission, target)
    citation_component = 0.35 if _normalize(submission.osha_citation) == _normalize(target.osha_citation) else 0.0
    severity_component = 0.10 if submission.severity == target.severity else 0.0
    evidence_component = 0.10 * min(1.0, _keyword_hit_ratio(submission.evidence, target.evidence_keywords) * 1.4)
    corrective_component = 0.15 * min(1.0, _keyword_hit_ratio(submission.corrective_action, target.corrective_keywords) * 1.4)
    return round(label_component + citation_component + severity_component + evidence_component + corrective_component, 4)


@dataclass
class MatchResult:
    submission_index: int
    target_id: str
    score: float


def _best_matches(
    submissions: Sequence[FindingSubmission], targets: Sequence[TargetFinding]
) -> Tuple[List[MatchResult], List[int]]:
    candidate_pairs: List[Tuple[float, int, str]] = []
    for sub_index, submission in enumerate(submissions):
        for target in targets:
            candidate_pairs.append((score_submission_against_target(submission, target), sub_index, target.finding_id))
    candidate_pairs.sort(reverse=True)

    used_submissions = set()
    used_targets = set()
    matches: List[MatchResult] = []
    for score, sub_index, target_id in candidate_pairs:
        if score <= 0.0 or sub_index in used_submissions or target_id in used_targets:
            continue
        matches.append(MatchResult(submission_index=sub_index, target_id=target_id, score=score))
        used_submissions.add(sub_index)
        used_targets.add(target_id)

    unmatched_submissions = [idx for idx in range(len(submissions)) if idx not in used_submissions]
    return matches, unmatched_submissions


def grade_task(
    task: InspectionTask,
    submissions: Sequence[FindingSubmission],
    steps_used: int,
    submitted_final: bool,
) -> Dict[str, object]:
    targets = task.target_findings
    matches, unmatched_submissions = _best_matches(submissions, targets)
    matched_score = sum(match.score for match in matches)
    base_score = matched_score / max(1, len(targets))

    duplicate_penalty = 0.0
    seen_pairs = set()
    for submission in submissions:
        pair = (_normalize(submission.hazard_label), _normalize(submission.osha_citation))
        if pair in seen_pairs:
            duplicate_penalty += 0.04
        seen_pairs.add(pair)

    hallucination_penalty = 0.06 * len(unmatched_submissions)
    excess_step_penalty = 0.02 * max(0, steps_used - len(targets) - 1)
    no_submit_penalty = 0.08 if not submitted_final else 0.0

    raw_total_score = round(base_score - duplicate_penalty - hallucination_penalty - excess_step_penalty - no_submit_penalty, 4)
    total_score = max(MIN_TASK_SCORE, min(MAX_TASK_SCORE, raw_total_score))
    per_target_scores = {target.finding_id: 0.0 for target in targets}
    for match in matches:
        per_target_scores[match.target_id] = round(match.score, 4)

    return {
        "score": total_score,
        "matched_targets": [match.target_id for match in matches],
        "per_target_scores": per_target_scores,
        "duplicate_penalty": round(duplicate_penalty, 4),
        "hallucination_penalty": round(hallucination_penalty, 4),
        "excess_step_penalty": round(excess_step_penalty, 4),
        "submitted_final": submitted_final,
        "target_count": len(targets),
        "submission_count": len(submissions),
        "matched_count": len(matches),
        "unmatched_submission_count": len(unmatched_submissions),
        "success": total_score >= 0.85,
    }
