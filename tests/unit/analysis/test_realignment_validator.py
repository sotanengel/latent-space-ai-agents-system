"""Tests for RealignmentValidator (FR-RA-001..010)."""

from __future__ import annotations

import torch

from latent_agents.analysis import (
    ConditionNumberVerifier,
    OrthogonalityVerifier,
    ProcrustesAligner,
    RealignmentValidator,
    RoundTripVerifier,
    SingularValueVerifier,
)
from latent_agents.schemas import Verdict


def _orth(d: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(d, d, generator=g)
    q, _ = torch.linalg.qr(a)
    return q


class TestProcrustesAligner:
    def test_recovers_orthogonal_rotation(self) -> None:
        d = 16
        x = torch.randn(64, d, generator=torch.Generator().manual_seed(1))
        rot = _orth(d, seed=2)
        y = x @ rot
        wa = ProcrustesAligner().fit(x, y)
        assert torch.allclose(wa, rot, atol=1e-4)

    def test_fit_is_deterministic(self) -> None:
        x = torch.randn(32, 8, generator=torch.Generator().manual_seed(3))
        y = torch.randn(32, 8, generator=torch.Generator().manual_seed(4))
        a = ProcrustesAligner().fit(x, y)
        b = ProcrustesAligner().fit(x, y)
        assert torch.equal(a, b)  # FR-RA-006

    def test_shape_is_square_hidden_dim(self) -> None:
        x = torch.randn(40, 12)
        y = torch.randn(40, 12)
        wa = ProcrustesAligner().fit(x, y)
        assert wa.shape == (12, 12)  # FR-RA-001


class TestConditionNumberVerifier:
    def test_well_conditioned_orthogonal_passes(self) -> None:
        v = ConditionNumberVerifier(max_kappa=1e4)
        r = v.verify(_orth(8))
        assert r.verdict == Verdict.PASS
        assert r.measured_value is not None
        assert r.measured_value < 10  # orthogonal -> kappa == 1

    def test_ill_conditioned_fails(self) -> None:
        d = 8
        u = _orth(d)
        s = torch.diag(torch.tensor([1.0] + [1e-6] * (d - 1)))
        v_ = _orth(d, seed=5)
        wa = u @ s @ v_.T
        v = ConditionNumberVerifier(max_kappa=1e4)
        r = v.verify(wa)
        assert r.verdict == Verdict.FAIL


class TestSingularValueVerifier:
    def test_minimum_singular_value_floor(self) -> None:
        v = SingularValueVerifier(min_singular=1e-6)
        assert v.verify(_orth(8)).verdict == Verdict.PASS

    def test_degenerate_fails(self) -> None:
        v = SingularValueVerifier(min_singular=1e-6)
        d = 6
        u = _orth(d)
        s = torch.diag(torch.tensor([1.0] * (d - 1) + [1e-12]))
        v_ = _orth(d, seed=9)
        wa = u @ s @ v_.T
        assert v.verify(wa).verdict == Verdict.FAIL


class TestOrthogonalityVerifier:
    def test_orthogonal_passes(self) -> None:
        v = OrthogonalityVerifier(tol=1e-4)
        assert v.verify(_orth(8)).verdict == Verdict.PASS

    def test_nonorthogonal_fails(self) -> None:
        v = OrthogonalityVerifier(tol=1e-4)
        wa = torch.eye(4) * 2
        assert v.verify(wa).verdict == Verdict.FAIL


class TestRoundTripVerifier:
    def test_orthogonal_roundtrip_passes(self) -> None:
        v = RoundTripVerifier(tol=1e-3)
        d = 8
        rot = _orth(d)
        x = torch.randn(16, d)
        assert v.verify((rot, x)).verdict == Verdict.PASS

    def test_near_singular_matrix_fails_roundtrip(self) -> None:
        v = RoundTripVerifier(tol=1e-3)
        d = 6
        u = _orth(d)
        # One vanishing singular value -> torch.linalg.inv raises and the
        # verifier classifies it as FAIL.
        s = torch.diag(torch.tensor([1.0] * (d - 1) + [0.0]))
        v_ = _orth(d, seed=11)
        wa = u @ s @ v_.T
        x = torch.randn(16, d)
        r = v.verify((wa, x))
        assert r.verdict == Verdict.FAIL


class TestRealignmentValidator:
    def test_validates_all_invariants(self) -> None:
        d = 8
        x = torch.randn(64, d, generator=torch.Generator().manual_seed(7))
        y = x @ _orth(d, seed=8)
        wa = ProcrustesAligner().fit(x, y)
        val = RealignmentValidator()
        results = val.validate(wa)
        assert all(r.verdict == Verdict.PASS for r in results)
        assert {r.test_id for r in results} >= {
            "FR-RA-001",
            "FR-RA-002",
            "FR-RA-003",
            "FR-RA-004",
        }
