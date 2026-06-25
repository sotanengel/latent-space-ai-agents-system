"""Tests for CrossModelAligner (FR-CM-004 projection, FR-CM-007 CKA)."""

from __future__ import annotations

import torch

from latent_agents.schemas import Verdict
from latent_agents.semantic import CKAVerifier, CrossModelAligner, LinearProjectionAligner


class TestLinearProjectionAligner:
    def test_projects_between_different_hidden_dims(self) -> None:
        src = torch.randn(64, 8)
        tgt = torch.randn(64, 16)
        a = LinearProjectionAligner().fit(src, tgt)
        assert a.shape == (8, 16)

    def test_projection_minimises_residual(self) -> None:
        src = torch.randn(128, 8)
        true_w = torch.randn(8, 16)
        tgt = src @ true_w + 0.01 * torch.randn(128, 16)
        a = LinearProjectionAligner().fit(src, tgt)
        pred = src @ a
        residual = (pred - tgt).norm() / tgt.norm()
        assert residual < 0.05


class TestCKAVerifier:
    def test_identical_features_high_cka(self) -> None:
        x = torch.randn(64, 12)
        v = CKAVerifier(min_cka=0.7)
        r = v.verify((x, x))
        assert r.verdict == Verdict.PASS
        assert r.measured_value is not None
        assert r.measured_value > 0.99

    def test_unrelated_features_low_cka(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(64, 12)
        y = torch.randn(64, 12)
        v = CKAVerifier(min_cka=0.7)
        r = v.verify((x, y))
        assert r.verdict == Verdict.FAIL
        assert r.measured_value is not None
        assert r.measured_value < 0.7


class TestCrossModelAligner:
    def test_align_returns_projection_when_dims_differ(self) -> None:
        src = torch.randn(64, 8)
        tgt = torch.randn(64, 16)
        a = CrossModelAligner()
        m = a.align(src, tgt)
        assert m.shape == (8, 16)

    def test_align_returns_orthogonal_when_dims_match(self) -> None:
        src = torch.randn(64, 12)
        tgt = torch.randn(64, 12)
        a = CrossModelAligner()
        m = a.align(src, tgt)
        assert m.shape == (12, 12)
