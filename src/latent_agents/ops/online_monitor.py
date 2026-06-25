"""OnlineMonitor (FR-OM-001..009).

Glues the drift detector + SLO window to the alert manager so a single
`observe(latent, latency_ms, error)` call is enough on the hot path.
Optional Prometheus exporter is wired in so SRE dashboards see the same
metrics the monitor reacts to.
"""

from __future__ import annotations

import torch

from .alert_manager import Alert, AlertManager, Severity
from .drift_detector import DriftDetector
from .prometheus_exporter import PrometheusExporter
from .slo import SLOWindow


class OnlineMonitor:
    def __init__(
        self,
        *,
        drift_detector: DriftDetector,
        alert_manager: AlertManager,
        slo: SLOWindow,
        prometheus: PrometheusExporter | None = None,
    ) -> None:
        self.drift = drift_detector
        self.alerts = alert_manager
        self.slo = slo
        self.prometheus = prometheus

    def observe(self, *, latent: torch.Tensor, latency_ms: float, error: bool) -> None:
        # SLO accounting
        self.slo.record(latency_ms=latency_ms, error=error)
        # Drift accounting
        score = self.drift.score(latent)
        if self.prometheus is not None:
            self.prometheus.inc("latent_observed_total")
            self.prometheus.set_gauge("drift_score", score)
            self.prometheus.set_gauge("latency_ms", latency_ms)
        if score > self.drift.threshold:
            self.alerts.fire(
                Alert(
                    severity=Severity.HIGH,
                    kind="drift",
                    message=f"drift score {score:.4f} > {self.drift.threshold}",
                )
            )
        if not self.slo.satisfied():
            self.alerts.fire(
                Alert(
                    severity=Severity.HIGH,
                    kind="slo",
                    message="SLO violation in sliding window",
                )
            )
