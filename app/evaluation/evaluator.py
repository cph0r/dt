from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EvaluationResult:
    score: float
    rationale: str
    passed: bool


@dataclass
class Evaluator:
    threshold: float = 0.75
    model_weight: float = 0.55
    eval_weight: float = 0.45

    def judge(self, question: str, answer: str, evidence: list[dict[str, Any]]) -> EvaluationResult:
        coverage = 1.0 if evidence else 0.45
        groundedness = 1.0 if any(chunk.get("source") for chunk in evidence) else 0.5
        completeness = 1.0 if len(answer) > 20 else 0.4
        score = round((coverage + groundedness + completeness) / 3.0, 3)
        passed = score >= self.threshold
        rationale = "evidence-backed" if passed else "insufficient grounding"
        return EvaluationResult(score=score, rationale=rationale, passed=passed)

    def adjusted_confidence(self, model_confidence: float, result: EvaluationResult) -> float:
        combined = self.model_weight * float(model_confidence) + self.eval_weight * result.score
        return round(max(0.0, min(1.0, combined)), 3)

    def should_fallback(self, adjusted_confidence: float, threshold: float) -> bool:
        return adjusted_confidence < threshold
