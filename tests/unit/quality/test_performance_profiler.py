"""Tests for PerformanceProfiler (FR-PF-001/002/009)."""

from __future__ import annotations

import time

import pytest

from latent_agents.quality import PerformanceProfiler


class TestPerformanceProfiler:
    def test_phase_records_wall_clock(self) -> None:
        p = PerformanceProfiler()
        with p.phase("inference"):
            time.sleep(0.01)
        assert p.summary().phases["inference"].wall_ns > 0

    def test_phases_accumulate(self) -> None:
        p = PerformanceProfiler()
        with p.phase("inference"):
            pass
        with p.phase("inference"):
            pass
        assert p.summary().phases["inference"].count == 2

    def test_record_tokens_supports_speedup_calc(self) -> None:
        p = PerformanceProfiler()
        p.record_tokens(baseline=1000, latent=300)
        s = p.summary()
        assert s.token_reduction == pytest.approx(0.7, abs=1e-3)

    def test_speedup_against_baseline(self) -> None:
        p = PerformanceProfiler()
        p.record_wall_clock(baseline_ns=1_000_000_000, latent_ns=500_000_000)
        s = p.summary()
        assert s.speedup == pytest.approx(2.0)

    def test_summary_phase_breakdown_percentages_sum_to_one(self) -> None:
        p = PerformanceProfiler()
        with p.phase("a"):
            time.sleep(0.001)
        with p.phase("b"):
            time.sleep(0.001)
        s = p.summary()
        total = sum(b.fraction for b in s.phases.values())
        assert 0.99 < total < 1.01


class TestProfileSummary:
    def test_to_dict_serialisable(self) -> None:
        p = PerformanceProfiler()
        with p.phase("x"):
            pass
        d = p.summary().to_dict()
        assert "phases" in d
        assert isinstance(d["phases"], dict)
