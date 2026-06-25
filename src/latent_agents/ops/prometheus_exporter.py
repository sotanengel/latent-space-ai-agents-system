"""Prometheus text-format exporter (NFR-OB-002).

We render the exposition format by hand so the package stays free of the
`prometheus_client` dependency. Drop-in compatible with a `/metrics`
endpoint served by any HTTP framework (FastAPI/Starlette/etc.).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

_MetricType = Literal["counter", "gauge"]


def _label_str(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return "{" + inner + "}"


class PrometheusExporter:
    def __init__(self) -> None:
        self._counters: dict[str, dict[tuple[tuple[str, str], ...], float]] = defaultdict(dict)
        self._gauges: dict[str, dict[tuple[tuple[str, str], ...], float]] = defaultdict(dict)

    def inc(self, name: str, *, by: float = 1.0, labels: dict[str, str] | None = None) -> None:
        key = tuple(sorted((labels or {}).items()))
        self._counters[name][key] = self._counters[name].get(key, 0.0) + by

    def set_gauge(self, name: str, value: float, *, labels: dict[str, str] | None = None) -> None:
        key = tuple(sorted((labels or {}).items()))
        self._gauges[name][key] = value

    def render(self) -> str:
        lines: list[str] = []
        for name, series in self._counters.items():
            lines.append(f"# TYPE {name} counter")
            for key, value in series.items():
                lines.append(f"{name}{_label_str(dict(key))} {value}")
        for name, series in self._gauges.items():
            lines.append(f"# TYPE {name} gauge")
            for key, value in series.items():
                lines.append(f"{name}{_label_str(dict(key))} {value}")
        return "\n".join(lines) + "\n"
