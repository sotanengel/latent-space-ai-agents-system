"""LatentFlowTracer (FR-PV-005 dependency DAG, lightweight spans).

A tiny span-and-DAG tracker for in-process flow tracing. It deliberately
mirrors the OpenTelemetry shape (`with span(...)`) so a future Phase 7
exporter can adapt these events into OTel spans without API churn here.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterator


@dataclass
class SpanEvent:
    name: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    start_ns: int = 0
    duration_ns: int | None = None
    error: str | None = None


class LatentFlowTracer:
    def __init__(self) -> None:
        self.events: list[SpanEvent] = []

    @contextmanager
    def span(
        self,
        name: str,
        *,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
    ) -> Iterator[SpanEvent]:
        ev = SpanEvent(
            name=name,
            inputs=list(inputs or []),
            outputs=list(outputs or []),
            start_ns=time.perf_counter_ns(),
        )
        self.events.append(ev)
        try:
            yield ev
        except Exception as exc:
            ev.error = str(exc)
            ev.duration_ns = time.perf_counter_ns() - ev.start_ns
            raise
        else:
            ev.duration_ns = time.perf_counter_ns() - ev.start_ns

    def dependency_dag(self) -> dict[str, set[str]]:
        """Return `{output -> set(inputs)}` aggregated across events."""
        dag: dict[str, set[str]] = {}
        for ev in self.events:
            for out in ev.outputs:
                dag.setdefault(out, set()).update(ev.inputs)
        return dag
