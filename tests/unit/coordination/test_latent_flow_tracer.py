"""Tests for LatentFlowTracer (FR-PV-005 dependency DAG, FR-PV-006 replay)."""

from __future__ import annotations

import pytest

from latent_agents.coordination import LatentFlowTracer


class TestLatentFlowTracer:
    def test_span_records_lifecycle(self) -> None:
        tr = LatentFlowTracer()
        with tr.span("agent-a.forward", inputs=["x"], outputs=["l1"]):
            pass
        events = tr.events
        assert len(events) == 1
        assert events[0].name == "agent-a.forward"
        assert events[0].duration_ns is not None

    def test_span_records_error(self) -> None:
        tr = LatentFlowTracer()
        with pytest.raises(RuntimeError), tr.span("bad-op"):
            raise RuntimeError("boom")
        assert tr.events[0].error == "boom"

    def test_dag_edges_from_inputs_outputs(self) -> None:
        tr = LatentFlowTracer()
        with tr.span("op-1", inputs=["x"], outputs=["l1"]):
            pass
        with tr.span("op-2", inputs=["l1"], outputs=["l2"]):
            pass
        dag = tr.dependency_dag()
        assert dag["l2"] == {"l1"}
        assert dag["l1"] == {"x"}
