"""Tests for OnlineMonitor (FR-OM-001..009)."""

from __future__ import annotations

import torch

from latent_agents.ops import (
    Alert,
    AlertManager,
    DriftDetector,
    OnlineMonitor,
    Severity,
    SLOWindow,
)


class _CollectAlerts:
    def __init__(self) -> None:
        self.alerts: list[Alert] = []

    def __call__(self, alert: Alert) -> None:
        self.alerts.append(alert)


class TestDriftDetector:
    def test_no_drift_for_in_distribution(self) -> None:
        torch.manual_seed(0)
        ref = torch.randn(256, 16)
        det = DriftDetector(threshold=0.5).fit(ref)
        score = det.score(torch.randn(32, 16))
        assert score < 0.5

    def test_drift_for_shifted_distribution(self) -> None:
        torch.manual_seed(1)
        ref = torch.randn(256, 16)
        det = DriftDetector(threshold=0.5).fit(ref)
        shifted = torch.randn(32, 16) + 5.0
        assert det.score(shifted) > 0.5


class TestSLOWindow:
    def test_slo_violation_detected(self) -> None:
        win = SLOWindow(max_latency_ms=100.0, max_error_rate=0.05, window=10)
        for _ in range(9):
            win.record(latency_ms=10.0, error=False)
        win.record(latency_ms=200.0, error=False)
        assert not win.satisfied()

    def test_slo_satisfied(self) -> None:
        win = SLOWindow(max_latency_ms=100.0, max_error_rate=0.05, window=10)
        for _ in range(10):
            win.record(latency_ms=20.0, error=False)
        assert win.satisfied()

    def test_error_rate_threshold(self) -> None:
        win = SLOWindow(max_latency_ms=1000.0, max_error_rate=0.1, window=10)
        for _ in range(8):
            win.record(latency_ms=10.0, error=False)
        win.record(latency_ms=10.0, error=True)
        win.record(latency_ms=10.0, error=True)
        assert not win.satisfied()


class TestAlertManager:
    def test_dispatches_to_sink(self) -> None:
        sink = _CollectAlerts()
        am = AlertManager(sinks=[sink])
        am.fire(Alert(severity=Severity.HIGH, kind="drift", message="x"))
        assert len(sink.alerts) == 1
        assert sink.alerts[0].kind == "drift"

    def test_dedup_within_window(self) -> None:
        sink = _CollectAlerts()
        am = AlertManager(sinks=[sink], dedup_window_seconds=10.0)
        am.fire(Alert(severity=Severity.HIGH, kind="drift", message="x"))
        am.fire(Alert(severity=Severity.HIGH, kind="drift", message="x"))
        assert len(sink.alerts) == 1


class TestOnlineMonitor:
    def test_drift_triggers_alert(self) -> None:
        torch.manual_seed(2)
        ref = torch.randn(256, 16)
        sink = _CollectAlerts()
        mon = OnlineMonitor(
            drift_detector=DriftDetector(threshold=0.5).fit(ref),
            alert_manager=AlertManager(sinks=[sink]),
            slo=SLOWindow(max_latency_ms=100.0, max_error_rate=0.05, window=5),
        )
        mon.observe(latent=torch.randn(8, 16) + 8.0, latency_ms=10.0, error=False)
        kinds = [a.kind for a in sink.alerts]
        assert "drift" in kinds

    def test_slo_violation_triggers_alert(self) -> None:
        torch.manual_seed(3)
        ref = torch.randn(64, 16)
        sink = _CollectAlerts()
        mon = OnlineMonitor(
            drift_detector=DriftDetector(threshold=0.5).fit(ref),
            alert_manager=AlertManager(sinks=[sink], dedup_window_seconds=0),
            slo=SLOWindow(max_latency_ms=100.0, max_error_rate=0.05, window=3),
        )
        for _ in range(3):
            mon.observe(latent=torch.randn(8, 16), latency_ms=500.0, error=False)
        assert any(a.kind == "slo" for a in sink.alerts)
