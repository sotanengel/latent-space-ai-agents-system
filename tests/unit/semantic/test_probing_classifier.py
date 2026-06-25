"""Tests for ProbingClassifier (FR-SM-002, FR-SM-004)."""

from __future__ import annotations

import torch

from latent_agents.schemas import Verdict
from latent_agents.semantic import ProbingAccuracyVerifier, ProbingClassifier


class TestProbingClassifier:
    def test_fits_and_predicts_separable_classes(self) -> None:
        torch.manual_seed(0)
        # Two clusters in 8D: easy linear probe.
        n = 64
        a = torch.randn(n, 8) + torch.tensor([3.0] * 8)
        b = torch.randn(n, 8) - torch.tensor([3.0] * 8)
        x = torch.cat([a, b])
        y = torch.tensor([0] * n + [1] * n)
        clf = ProbingClassifier(num_classes=2)
        clf.fit(x, y, epochs=50)
        pred = clf.predict(x)
        acc = float((pred == y).float().mean().item())
        assert acc > 0.9

    def test_score_returns_accuracy_in_unit_interval(self) -> None:
        torch.manual_seed(1)
        x = torch.randn(40, 4)
        y = torch.tensor([0] * 20 + [1] * 20)
        clf = ProbingClassifier(num_classes=2)
        clf.fit(x, y, epochs=10)
        s = clf.score(x, y)
        assert 0.0 <= s <= 1.0


class TestProbingAccuracyVerifier:
    def test_pass_above_threshold(self) -> None:
        torch.manual_seed(2)
        n = 64
        a = torch.randn(n, 8) + 3
        b = torch.randn(n, 8) - 3
        x = torch.cat([a, b])
        y = torch.tensor([0] * n + [1] * n)
        v = ProbingAccuracyVerifier(min_accuracy=0.85, epochs=50, num_classes=2)
        r = v.verify((x, y))
        assert r.verdict == Verdict.PASS

    def test_fail_below_threshold(self) -> None:
        torch.manual_seed(3)
        x = torch.randn(40, 4)
        y = torch.randint(0, 2, (40,))
        v = ProbingAccuracyVerifier(min_accuracy=0.99, epochs=5, num_classes=2)
        r = v.verify((x, y))
        assert r.verdict == Verdict.FAIL
