"""Tests for the Prometheus text-format exporter (NFR-OB-002)."""

from __future__ import annotations

from latent_agents.ops import PrometheusExporter


class TestPrometheusExporter:
    def test_counter_renders(self) -> None:
        e = PrometheusExporter()
        e.inc("latent_observed_total")
        text = e.render()
        assert "# TYPE latent_observed_total counter" in text
        assert "latent_observed_total 1.0" in text

    def test_gauge_renders(self) -> None:
        e = PrometheusExporter()
        e.set_gauge("drift_score", 0.42)
        text = e.render()
        assert "# TYPE drift_score gauge" in text
        assert "drift_score 0.42" in text

    def test_labels_emit_per_series(self) -> None:
        e = PrometheusExporter()
        e.inc("requests_total", labels={"agent": "a", "model": "tiny"})
        e.inc("requests_total", labels={"agent": "b", "model": "tiny"})
        text = e.render()
        assert 'requests_total{agent="a",model="tiny"} 1.0' in text
        assert 'requests_total{agent="b",model="tiny"} 1.0' in text
