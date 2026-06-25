"""Tests for the TensorInspector verifier suite (FR-LG-001/002/004/008/012, FR-LG-003)."""

from __future__ import annotations

import torch

from latent_agents.analysis import (
    BatchIndependenceVerifier,
    DtypeVerifier,
    NanInfVerifier,
    NormStabilityVerifier,
    ShapeVerifier,
    SparsityVerifier,
)
from latent_agents.schemas import Verdict


class TestShapeVerifier:
    def test_pass_on_matching_hidden_dim(self) -> None:
        v = ShapeVerifier(expected_hidden_dim=8)
        r = v.verify(torch.zeros(2, 4, 8))
        assert r.verdict == Verdict.PASS
        assert r.test_id == "FR-LG-001"

    def test_fail_on_mismatched_hidden_dim(self) -> None:
        v = ShapeVerifier(expected_hidden_dim=16)
        r = v.verify(torch.zeros(2, 4, 8))
        assert r.verdict == Verdict.FAIL

    def test_fail_on_wrong_rank(self) -> None:
        v = ShapeVerifier(expected_hidden_dim=8)
        r = v.verify(torch.zeros(2, 8))  # 2D
        assert r.verdict == Verdict.FAIL


class TestDtypeVerifier:
    def test_pass_when_dtype_matches(self) -> None:
        v = DtypeVerifier(expected_dtype=torch.float16)
        r = v.verify(torch.zeros(1, 1, 1, dtype=torch.float16))
        assert r.verdict == Verdict.PASS

    def test_fail_when_dtype_mismatches(self) -> None:
        v = DtypeVerifier(expected_dtype=torch.float16)
        r = v.verify(torch.zeros(1, 1, 1, dtype=torch.float32))
        assert r.verdict == Verdict.FAIL


class TestNanInfVerifier:
    def test_pass_on_clean_tensor(self) -> None:
        v = NanInfVerifier()
        r = v.verify(torch.randn(2, 3, 4))
        assert r.verdict == Verdict.PASS

    def test_fail_on_nan(self) -> None:
        v = NanInfVerifier()
        t = torch.tensor([[[1.0, float("nan"), 3.0]]])
        r = v.verify(t)
        assert r.verdict == Verdict.FAIL

    def test_fail_on_inf(self) -> None:
        v = NanInfVerifier()
        t = torch.tensor([[[1.0, float("inf"), 3.0]]])
        r = v.verify(t)
        assert r.verdict == Verdict.FAIL


class TestNormStabilityVerifier:
    def test_pass_within_band(self) -> None:
        v = NormStabilityVerifier()
        h0 = torch.ones(1, 4, 8)
        ht = torch.ones(1, 4, 8) * 1.5  # ratio 1.5, within [0.5, 2.0]
        r = v.verify((h0, ht))
        assert r.verdict == Verdict.PASS
        assert r.measured_value is not None
        assert abs(r.measured_value - 1.5) < 1e-5

    def test_fail_on_explosion(self) -> None:
        v = NormStabilityVerifier()
        h0 = torch.ones(1, 4, 8)
        ht = torch.ones(1, 4, 8) * 10  # ratio 10, > 2.0
        r = v.verify((h0, ht))
        assert r.verdict == Verdict.FAIL

    def test_fail_on_collapse(self) -> None:
        v = NormStabilityVerifier()
        h0 = torch.ones(1, 4, 8)
        ht = torch.ones(1, 4, 8) * 0.1  # ratio 0.1, < 0.5
        r = v.verify((h0, ht))
        assert r.verdict == Verdict.FAIL


class TestSparsityVerifier:
    def test_pass_within_band(self) -> None:
        v = SparsityVerifier(expected=0.5, tolerance=0.1)
        t = torch.zeros(2, 4, 10)
        t[..., :5] = 1.0  # 50% sparse
        r = v.verify(t)
        assert r.verdict == Verdict.PASS

    def test_fail_when_too_sparse(self) -> None:
        v = SparsityVerifier(expected=0.5, tolerance=0.1)
        r = v.verify(torch.zeros(2, 4, 10))  # 100% sparse
        assert r.verdict == Verdict.FAIL


class TestBatchIndependenceVerifier:
    def test_pass_on_independent_samples(self) -> None:
        torch.manual_seed(0)
        v = BatchIndependenceVerifier(threshold=0.5)
        batch = torch.randn(8, 4, 16)
        r = v.verify(batch)
        assert r.verdict == Verdict.PASS

    def test_fail_on_identical_samples(self) -> None:
        v = BatchIndependenceVerifier(threshold=0.5)
        row = torch.randn(1, 4, 16)
        batch = row.expand(8, 4, 16).contiguous()
        r = v.verify(batch)
        assert r.verdict == Verdict.FAIL
