"""LLM-as-Judge orchestrator (FR-TA-001..003).

Implements:
- `Judge` protocol: any callable that returns (verdict, score) for a (sample, reference) pair.
- `LLMJudgeOrchestrator`: ensemble of judges with quorum-based decision and Cohen's Kappa
  computation against reference labels.

The orchestrator is provider-agnostic — concrete LLM-backed judges live
outside this module so CI can substitute deterministic mock judges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol


class JudgeVerdict(StrEnum):
    EQUIVALENT = "equivalent"
    DIFFERENT = "different"


class Judge(Protocol):
    name: str

    def evaluate(self, sample: str, reference: str) -> tuple[JudgeVerdict, float]: ...


@dataclass(frozen=True)
class EnsembleResult:
    verdict: JudgeVerdict
    mean_score: float
    individual: list[tuple[str, JudgeVerdict, float]] = field(default_factory=list)


class LLMJudgeOrchestrator:
    """Run a fixed set of judges and aggregate by quorum."""

    def __init__(self, *, judges: list[Judge], quorum: int = 2) -> None:
        if not judges:
            raise ValueError("LLMJudgeOrchestrator requires at least one judge")
        if quorum < 1 or quorum > len(judges):
            raise ValueError(f"quorum must be in [1, {len(judges)}], got {quorum}")
        self.judges = judges
        self.quorum = quorum

    def evaluate(self, sample: str, reference: str) -> EnsembleResult:
        per_judge: list[tuple[str, JudgeVerdict, float]] = []
        for j in self.judges:
            v, s = j.evaluate(sample, reference)
            per_judge.append((j.name, v, s))
        eq_votes = sum(1 for _, v, _ in per_judge if v == JudgeVerdict.EQUIVALENT)
        verdict = JudgeVerdict.EQUIVALENT if eq_votes >= self.quorum else JudgeVerdict.DIFFERENT
        mean = sum(s for _, _, s in per_judge) / len(per_judge)
        return EnsembleResult(verdict=verdict, mean_score=mean, individual=per_judge)

    def cohens_kappa(
        self,
        pairs: list[tuple[str, str]],
        reference_labels: list[JudgeVerdict],
    ) -> float:
        """Compute Cohen's kappa between ensemble verdict and reference labels."""
        if len(pairs) != len(reference_labels):
            raise ValueError("pairs and reference_labels must align")
        n = len(pairs)
        if n == 0:
            return 0.0
        ours = [self.evaluate(s, r).verdict for s, r in pairs]
        agree = sum(1 for o, r in zip(ours, reference_labels, strict=True) if o == r)
        p_o = agree / n
        # Per-class marginals
        eq_ours = sum(1 for v in ours if v == JudgeVerdict.EQUIVALENT) / n
        eq_ref = sum(1 for v in reference_labels if v == JudgeVerdict.EQUIVALENT) / n
        p_e = eq_ours * eq_ref + (1 - eq_ours) * (1 - eq_ref)
        if p_e == 1.0:
            # Perfect chance agreement — by convention return 1.0 when also fully agreeing.
            return 1.0 if p_o == 1.0 else 0.0
        return (p_o - p_e) / (1 - p_e)
