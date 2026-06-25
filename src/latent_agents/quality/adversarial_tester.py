"""AdversarialTester suite (FR-RB-001..007, FR-RB-009/011).

Implements detection-side and attack-side primitives:
- `NoiseRobustnessVerifier` (FR-RB-001): bounded output change under noise.
- `PoisoningDetector` (FR-RB-002): Mahalanobis-distance outlier rejection.
- `StatisticalAnomalyDetector` (FR-RB-004): per-feature z-score gate.
- `PromptInjectionDetector` (FR-RB-006): pattern-based string check.
- `FGSMAttack` (FR-RB-003): gradient-sign perturbation.
- `Watermarker` (FR-RB-007): low-amplitude key-derived watermark.
- `AdversarialTester` orchestrates the bundle for a quick "is this latent
  suspicious" report.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from typing import Any

import torch

from latent_agents.schemas import Verdict
from latent_agents.verifier import BaseVerifier, register


@register
class NoiseRobustnessVerifier(
    BaseVerifier[tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor], float]]
):
    """FR-RB-001: relative output change under additive Gaussian noise."""

    test_id = "FR-RB-001"

    def __init__(self, max_relative_change: float = 0.1) -> None:
        super().__init__()
        self.threshold = max_relative_change

    def _evaluate(
        self,
        target: tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor], float],
    ) -> tuple[Verdict, str | None]:
        x, fn, sigma = target
        clean = fn(x)
        noisy = fn(x + sigma * torch.randn_like(x))
        denom = clean.norm().clamp_min(1e-30)
        change = float(((noisy - clean).norm() / denom).item())
        if change > float(self.threshold):
            return Verdict.FAIL, f"relative change {change:.4f} > {self.threshold}"
        return Verdict.PASS, None

    def _measured(
        self,
        target: tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor], float],
    ) -> float:
        x, fn, sigma = target
        clean = fn(x)
        noisy = fn(x + sigma * torch.randn_like(x))
        denom = clean.norm().clamp_min(1e-30)
        return float(((noisy - clean).norm() / denom).item())


class PoisoningDetector:
    """Mahalanobis-distance gate over a known-clean reference set (FR-RB-002)."""

    def __init__(self, threshold: float = 25.0) -> None:
        self.threshold = threshold
        self._mean: torch.Tensor | None = None
        self._inv_cov: torch.Tensor | None = None

    def fit(self, clean: torch.Tensor) -> PoisoningDetector:
        if clean.ndim != 2:
            raise ValueError(f"clean must be 2D, got {tuple(clean.shape)}")
        x = clean.to(torch.float64)
        self._mean = x.mean(dim=0)
        centered = x - self._mean
        cov = centered.t() @ centered / max(x.shape[0] - 1, 1)
        cov += 1e-6 * torch.eye(cov.shape[0], dtype=cov.dtype)
        self._inv_cov = torch.linalg.inv(cov)
        return self

    def _distance(self, sample: torch.Tensor) -> float:
        if self._mean is None or self._inv_cov is None:
            raise RuntimeError("call fit() before is_poisoned()")
        v = sample.to(torch.float64) - self._mean
        d2 = (v @ self._inv_cov * v).sum(dim=-1)
        return float(d2.max().item())

    def is_poisoned(self, sample: torch.Tensor) -> bool:
        return self._distance(sample) > self.threshold


class StatisticalAnomalyDetector:
    """Per-feature z-score gate (FR-RB-004)."""

    def __init__(self, z: float = 3.0) -> None:
        self.z = z
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

    def fit(self, clean: torch.Tensor) -> StatisticalAnomalyDetector:
        self._mean = clean.mean(dim=0)
        self._std = clean.std(dim=0, unbiased=False).clamp_min(1e-30)
        return self

    def is_anomaly(self, sample: torch.Tensor) -> bool:
        if self._mean is None or self._std is None:
            raise RuntimeError("call fit() before is_anomaly()")
        zscore = ((sample - self._mean) / self._std).abs()
        return bool(zscore.max().item() > self.z)


class PromptInjectionDetector:
    """Pattern-based prompt-injection detection (FR-RB-006).

    Patterns cover the most-cited injection families: instruction overrides,
    persona resets, jailbreak preambles, system-prompt extraction.
    """

    _PATTERNS = (
        re.compile(r"ignore (?:all )?previous instructions", re.I),
        re.compile(r"disregard (?:any|all) (?:prior|above)", re.I),
        re.compile(r"system prompt", re.I),
        re.compile(r"BEGIN SYSTEM", re.I),
        re.compile(r"you are (?:now )?jailbroken", re.I),
        re.compile(r"DAN mode", re.I),
    )

    def contains_injection(self, text: str) -> bool:
        return any(p.search(text) for p in self._PATTERNS)


class FGSMAttack:
    """Single-step FGSM perturbation (FR-RB-003)."""

    def __init__(self, epsilon: float = 0.03) -> None:
        self.epsilon = epsilon

    def attack(self, x: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        return (x + self.epsilon * grad.sign()).detach()


class Watermarker:
    """Key-derived low-amplitude watermark on continuous tensors (FR-RB-007)."""

    def __init__(self, *, secret_key: bytes, strength: float = 0.01) -> None:
        self._key = secret_key
        self._strength = strength

    def _pattern(self, shape: tuple[int, ...]) -> torch.Tensor:
        seed = int.from_bytes(hashlib.sha256(self._key).digest()[:8], "big")
        g = torch.Generator().manual_seed(seed & 0x7FFFFFFF)
        p = torch.randn(*shape, generator=g)
        normed: torch.Tensor = p / p.norm().clamp_min(1e-30)
        return normed

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return x + self._strength * self._pattern(tuple(x.shape))

    def detect(self, x: torch.Tensor) -> float:
        pattern = self._pattern(tuple(x.shape))
        x_norm = x / x.norm().clamp_min(1e-30)
        return float((x_norm * pattern).sum().abs().item())


class AdversarialTester:
    """Bundle the detectors and produce a quick report on a candidate latent."""

    def __init__(
        self,
        *,
        poisoning_threshold: float = 25.0,
        anomaly_z: float = 3.0,
    ) -> None:
        self._poison = PoisoningDetector(threshold=poisoning_threshold)
        self._anomaly = StatisticalAnomalyDetector(z=anomaly_z)
        self._injection = PromptInjectionDetector()

    def fit(self, clean: torch.Tensor) -> AdversarialTester:
        self._poison.fit(clean)
        self._anomaly.fit(clean)
        return self

    def run(self, sample: torch.Tensor, *, decoded_text: str = "") -> dict[str, Any]:
        return {
            "poisoning": self._poison.is_poisoned(sample),
            "anomaly": self._anomaly.is_anomaly(sample),
            "prompt_injection": self._injection.contains_injection(decoded_text),
        }
