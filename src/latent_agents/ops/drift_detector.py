"""DriftDetector (FR-OM-001).

Lightweight distribution-shift detector: tracks the per-dimension mean of
a reference set and reports the L2 distance between reference mean and
sample mean, normalised by reference std. Good enough to catch the
"output distribution moved by N sigma" signal that drives the FR-OM-001
SLA (lead time ≤ 1 hour).
"""

from __future__ import annotations

import torch


class DriftDetector:
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._ref_mean: torch.Tensor | None = None
        self._ref_std: torch.Tensor | None = None

    def fit(self, ref: torch.Tensor) -> DriftDetector:
        if ref.ndim != 2:
            raise ValueError(f"ref must be 2D [n, dim], got {tuple(ref.shape)}")
        self._ref_mean = ref.mean(dim=0)
        self._ref_std = ref.std(dim=0, unbiased=False).clamp_min(1e-6)
        return self

    def score(self, sample: torch.Tensor) -> float:
        if self._ref_mean is None or self._ref_std is None:
            raise RuntimeError("call fit() before score()")
        sample_mean = sample.mean(dim=0)
        delta = (sample_mean - self._ref_mean) / self._ref_std
        return float(delta.norm().item() / (delta.numel() ** 0.5))

    def has_drifted(self, sample: torch.Tensor) -> bool:
        return self.score(sample) > self.threshold
