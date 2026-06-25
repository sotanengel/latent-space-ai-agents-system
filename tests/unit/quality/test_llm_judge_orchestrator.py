"""Tests for LLMJudgeOrchestrator ensemble (FR-TA-001..003)."""

from __future__ import annotations

import pytest

from latent_agents.quality import Judge, JudgeVerdict, LLMJudgeOrchestrator


class _ConstJudge(Judge):
    def __init__(self, name: str, verdict: JudgeVerdict, score: float = 1.0) -> None:
        self.name = name
        self._v = verdict
        self._s = score

    def evaluate(self, sample: str, reference: str) -> tuple[JudgeVerdict, float]:
        return self._v, self._s


class TestLLMJudgeOrchestrator:
    def test_unanimous_pass(self) -> None:
        orch = LLMJudgeOrchestrator(
            judges=[
                _ConstJudge("a", JudgeVerdict.EQUIVALENT, 1.0),
                _ConstJudge("b", JudgeVerdict.EQUIVALENT, 0.9),
                _ConstJudge("c", JudgeVerdict.EQUIVALENT, 0.95),
            ],
            quorum=2,
        )
        result = orch.evaluate("a", "b")
        assert result.verdict == JudgeVerdict.EQUIVALENT
        assert result.mean_score > 0.9
        assert len(result.individual) == 3

    def test_quorum_required_for_pass(self) -> None:
        orch = LLMJudgeOrchestrator(
            judges=[
                _ConstJudge("a", JudgeVerdict.EQUIVALENT),
                _ConstJudge("b", JudgeVerdict.DIFFERENT),
                _ConstJudge("c", JudgeVerdict.DIFFERENT),
            ],
            quorum=2,
        )
        result = orch.evaluate("a", "b")
        assert result.verdict == JudgeVerdict.DIFFERENT

    def test_requires_at_least_one_judge(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            LLMJudgeOrchestrator(judges=[], quorum=1)

    def test_cohens_kappa_against_reference_labels(self) -> None:
        orch = LLMJudgeOrchestrator(
            judges=[_ConstJudge("a", JudgeVerdict.EQUIVALENT)],
            quorum=1,
        )
        # Three samples where reference labels are all "equivalent" and judge agrees.
        pairs = [("x", "x"), ("y", "y"), ("z", "z")]
        labels = [JudgeVerdict.EQUIVALENT] * 3
        kappa = orch.cohens_kappa(pairs, labels)
        # Trivially perfect agreement but degenerate (single class) -> 1.0.
        assert kappa == 1.0
