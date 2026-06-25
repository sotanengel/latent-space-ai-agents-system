"""Alert delivery (FR-OM-002 / FR-OM-008).

Tiny in-process alert manager with deduplication. Sinks are any callable
that accepts an `Alert`; real integrations (Slack, PagerDuty, OpsGenie)
just supply a sink. Rollback automation is *advisory* here — we expose
a `should_rollback()` helper that callers can react to.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum


class Severity(StrEnum):
    LOW = "low"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Alert:
    severity: Severity
    kind: str
    message: str


class AlertManager:
    def __init__(
        self,
        *,
        sinks: list[Callable[[Alert], None]] | None = None,
        dedup_window_seconds: float = 60.0,
    ) -> None:
        self.sinks: list[Callable[[Alert], None]] = list(sinks or [])
        self.dedup_window_seconds = dedup_window_seconds
        self._last_seen: dict[tuple[Severity, str, str], float] = {}

    def fire(self, alert: Alert) -> bool:
        """Dispatch to sinks unless an identical alert fired within the dedup window."""
        now = time.monotonic()
        key = (alert.severity, alert.kind, alert.message)
        last = self._last_seen.get(key)
        if last is not None and (now - last) < self.dedup_window_seconds:
            return False
        self._last_seen[key] = now
        for sink in self.sinks:
            sink(alert)
        return True

    def should_rollback(self) -> bool:
        return any(sev == Severity.CRITICAL for (sev, _, _) in self._last_seen)
