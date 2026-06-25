"""PerformanceProfiler (FR-PF-001 token reduction, FR-PF-002 speedup,
FR-PF-009 latency breakdown).

Lightweight wall-clock profiler with a `with` interface. Captures phase
durations and aggregates a percentage breakdown. Token-reduction and
speedup metrics are recorded explicitly (caller supplies baseline) since
the comparison subject is application-specific.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterator


@dataclass
class PhaseBreakdown:
    count: int
    wall_ns: int
    fraction: float


@dataclass
class ProfileSummary:
    phases: dict[str, PhaseBreakdown] = field(default_factory=dict)
    token_reduction: float | None = None
    speedup: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "phases": {
                k: {"count": v.count, "wall_ns": v.wall_ns, "fraction": v.fraction}
                for k, v in self.phases.items()
            },
            "token_reduction": self.token_reduction,
            "speedup": self.speedup,
        }


class PerformanceProfiler:
    def __init__(self) -> None:
        self._phase_wall: dict[str, int] = {}
        self._phase_count: dict[str, int] = {}
        self._token_reduction: float | None = None
        self._speedup: float | None = None

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed = time.perf_counter_ns() - start
            self._phase_wall[name] = self._phase_wall.get(name, 0) + elapsed
            self._phase_count[name] = self._phase_count.get(name, 0) + 1

    def record_tokens(self, *, baseline: int, latent: int) -> None:
        if baseline <= 0:
            raise ValueError("baseline must be > 0")
        self._token_reduction = 1.0 - (latent / baseline)

    def record_wall_clock(self, *, baseline_ns: int, latent_ns: int) -> None:
        if latent_ns <= 0:
            raise ValueError("latent_ns must be > 0")
        self._speedup = baseline_ns / latent_ns

    def summary(self) -> ProfileSummary:
        total = max(sum(self._phase_wall.values()), 1)
        phases = {
            name: PhaseBreakdown(
                count=self._phase_count[name],
                wall_ns=wall,
                fraction=wall / total,
            )
            for name, wall in self._phase_wall.items()
        }
        return ProfileSummary(
            phases=phases,
            token_reduction=self._token_reduction,
            speedup=self._speedup,
        )
