"""Tests for the failure-injection toolkit (FR-TA-005)."""

from __future__ import annotations

import pytest
import torch

from latent_agents.quality import (
    ChaosInjector,
    FailureMode,
    delay_call,
    flip_bits,
)


class TestFlipBits:
    def test_flips_at_least_one_bit(self) -> None:
        t = torch.randn(64)
        flipped = flip_bits(t, n=4, seed=0)
        assert not torch.equal(t, flipped)

    def test_seed_reproducibility(self) -> None:
        t = torch.randn(32)
        a = flip_bits(t, n=2, seed=1)
        b = flip_bits(t, n=2, seed=1)
        assert torch.equal(a, b)


class TestDelayCall:
    def test_adds_delay(self) -> None:
        import time as _t

        called = []

        def f() -> str:
            called.append(_t.perf_counter())
            return "ok"

        t0 = _t.perf_counter()
        out = delay_call(f, delay_seconds=0.02)()
        t1 = _t.perf_counter()
        assert out == "ok"
        assert t1 - t0 >= 0.02


class TestChaosInjector:
    def test_random_kill_when_active(self) -> None:
        ch = ChaosInjector(seed=0)
        ch.activate(FailureMode.AGENT_KILL, probability=1.0)
        with pytest.raises(RuntimeError, match="injected agent kill"):
            ch.maybe_fail()

    def test_no_failure_when_inactive(self) -> None:
        ch = ChaosInjector(seed=0)
        ch.maybe_fail()  # should not raise

    def test_corrupt_tensor(self) -> None:
        ch = ChaosInjector(seed=0)
        ch.activate(FailureMode.TENSOR_CORRUPT, probability=1.0)
        t = torch.zeros(8)
        out = ch.maybe_corrupt(t)
        assert not torch.equal(out, t)

    def test_failure_modes_enumerable(self) -> None:
        modes = list(FailureMode)
        assert FailureMode.AGENT_KILL in modes
        assert len(modes) >= 3
