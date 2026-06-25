"""SLO sliding-window tracker (FR-OM-002)."""

from __future__ import annotations

from collections import deque


class SLOWindow:
    def __init__(self, *, max_latency_ms: float, max_error_rate: float, window: int) -> None:
        self.max_latency_ms = max_latency_ms
        self.max_error_rate = max_error_rate
        self._latencies: deque[float] = deque(maxlen=window)
        self._errors: deque[bool] = deque(maxlen=window)

    def record(self, *, latency_ms: float, error: bool) -> None:
        self._latencies.append(latency_ms)
        self._errors.append(error)

    def satisfied(self) -> bool:
        if not self._latencies:
            return True
        if max(self._latencies) > self.max_latency_ms:
            return False
        err_rate = sum(self._errors) / len(self._errors)
        return err_rate <= self.max_error_rate
