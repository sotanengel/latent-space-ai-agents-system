"""TensorInspector verifiers (FR-LG-001/002/003/004/005/008/012).

Each verifier is a small class that consumes a torch tensor (or in some
cases a pair of tensors) and returns a `VerificationResult`. They are all
registered in the verifier registry so they can be discovered by FR-ID.
"""

from __future__ import annotations

from typing import Any

import torch

from latent_agents.schemas import Verdict
from latent_agents.verifier import BaseVerifier, register

_DTYPE_TO_STR: dict[torch.dtype, str] = {
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.int8: "int8",
}


@register
class ShapeVerifier(BaseVerifier[torch.Tensor]):
    """FR-LG-001: hidden_states shape must be `[batch, seq_len, hidden_dim]`."""

    test_id = "FR-LG-001"

    def __init__(self, expected_hidden_dim: int) -> None:
        super().__init__()
        self.expected_hidden_dim = expected_hidden_dim
        self.threshold = expected_hidden_dim

    def _evaluate(self, target: torch.Tensor) -> tuple[Verdict, str | None]:
        if target.ndim != 3:
            return Verdict.FAIL, f"expected rank 3, got {target.ndim}"
        if target.shape[-1] != self.expected_hidden_dim:
            return (
                Verdict.FAIL,
                f"hidden_dim={target.shape[-1]} != expected {self.expected_hidden_dim}",
            )
        return Verdict.PASS, None

    def _measured(self, target: torch.Tensor) -> Any:
        return tuple(target.shape)


@register
class DtypeVerifier(BaseVerifier[torch.Tensor]):
    """FR-LG-002: dtype must equal the expected dtype and not be mixed."""

    test_id = "FR-LG-002"

    def __init__(self, expected_dtype: torch.dtype) -> None:
        super().__init__()
        self.expected_dtype = expected_dtype
        self.threshold = _DTYPE_TO_STR.get(expected_dtype, str(expected_dtype))

    def _evaluate(self, target: torch.Tensor) -> tuple[Verdict, str | None]:
        if target.dtype != self.expected_dtype:
            return (
                Verdict.FAIL,
                f"dtype={target.dtype}, expected {self.expected_dtype}",
            )
        return Verdict.PASS, None

    def _measured(self, target: torch.Tensor) -> Any:
        return _DTYPE_TO_STR.get(target.dtype, str(target.dtype))


@register
class NanInfVerifier(BaseVerifier[torch.Tensor]):
    """FR-LG-004: hidden_states must not contain NaN or Inf."""

    test_id = "FR-LG-004"

    def _evaluate(self, target: torch.Tensor) -> tuple[Verdict, str | None]:
        if torch.isnan(target).any().item():
            return Verdict.FAIL, "NaN detected"
        if torch.isinf(target).any().item():
            return Verdict.FAIL, "Inf detected"
        return Verdict.PASS, None


@register
class NormStabilityVerifier(BaseVerifier[tuple[torch.Tensor, torch.Tensor]]):
    """FR-LG-003: 0.5 <= ||h_t|| / ||h_0|| <= 2.0."""

    test_id = "FR-LG-003"
    threshold: tuple[float, float] = (0.5, 2.0)

    def __init__(self, band: tuple[float, float] = (0.5, 2.0)) -> None:
        super().__init__()
        self.threshold = band

    def _ratio(self, target: tuple[torch.Tensor, torch.Tensor]) -> float:
        h0, ht = target
        n0 = torch.linalg.vector_norm(h0).item()
        nt = torch.linalg.vector_norm(ht).item()
        if n0 == 0:
            return float("inf") if nt > 0 else 1.0
        return float(nt / n0)

    def _evaluate(self, target: tuple[torch.Tensor, torch.Tensor]) -> tuple[Verdict, str | None]:
        lo, hi = self.threshold
        ratio = self._ratio(target)
        if lo <= ratio <= hi:
            return Verdict.PASS, None
        return Verdict.FAIL, f"norm ratio {ratio:.4f} outside [{lo}, {hi}]"

    def _measured(self, target: tuple[torch.Tensor, torch.Tensor]) -> Any:
        return self._ratio(target)


@register
class DistributionVerifier(BaseVerifier[torch.Tensor]):
    """FR-LG-005: distribution statistics within `sigma` of expected mean/std."""

    test_id = "FR-LG-005"

    def __init__(
        self,
        expected_mean: float = 0.0,
        expected_std: float = 1.0,
        sigma: float = 3.0,
    ) -> None:
        super().__init__()
        self.expected_mean = expected_mean
        self.expected_std = expected_std
        self.sigma = sigma
        self.threshold = (expected_mean, expected_std, sigma)

    def _evaluate(self, target: torch.Tensor) -> tuple[Verdict, str | None]:
        mean = float(target.mean().item())
        std = float(target.std(unbiased=False).item())
        if abs(mean - self.expected_mean) > self.sigma * self.expected_std:
            return Verdict.FAIL, f"mean {mean:.4f} > {self.sigma}-sigma from {self.expected_mean}"
        if abs(std - self.expected_std) > self.sigma * self.expected_std:
            return Verdict.FAIL, f"std {std:.4f} > {self.sigma}-sigma from {self.expected_std}"
        return Verdict.PASS, None

    def _measured(self, target: torch.Tensor) -> Any:
        return {
            "mean": float(target.mean().item()),
            "std": float(target.std(unbiased=False).item()),
        }


@register
class SparsityVerifier(BaseVerifier[torch.Tensor]):
    """FR-LG-008: sparsity (fraction of zeros) must lie within tolerance band."""

    test_id = "FR-LG-008"

    def __init__(self, expected: float, tolerance: float = 0.1) -> None:
        super().__init__()
        self.expected = expected
        self.tolerance = tolerance
        self.threshold = (expected - tolerance, expected + tolerance)

    def _evaluate(self, target: torch.Tensor) -> tuple[Verdict, str | None]:
        sparsity = float((target == 0).float().mean().item())
        if abs(sparsity - self.expected) > self.tolerance:
            return (
                Verdict.FAIL,
                f"sparsity {sparsity:.3f} outside [{self.expected - self.tolerance:.3f}, "
                f"{self.expected + self.tolerance:.3f}]",
            )
        return Verdict.PASS, None

    def _measured(self, target: torch.Tensor) -> Any:
        return float((target == 0).float().mean().item())


@register
class BatchIndependenceVerifier(BaseVerifier[torch.Tensor]):
    """FR-LG-012: batch samples must be independent (low pairwise cosine).

    We flatten per-sample, compute mean off-diagonal cosine similarity, and
    fail if it exceeds `threshold`.
    """

    test_id = "FR-LG-012"

    def __init__(self, threshold: float = 0.95) -> None:
        super().__init__()
        self.threshold = threshold

    def _mean_offdiag_cos(self, target: torch.Tensor) -> float:
        b = target.shape[0]
        if b < 2:
            return 0.0
        flat = target.reshape(b, -1).float()
        norm = flat / (flat.norm(dim=1, keepdim=True) + 1e-12)
        sim = norm @ norm.t()
        mask = ~torch.eye(b, dtype=torch.bool)
        return float(sim[mask].abs().mean().item())

    def _evaluate(self, target: torch.Tensor) -> tuple[Verdict, str | None]:
        cos = self._mean_offdiag_cos(target)
        if cos > self.threshold:
            return Verdict.FAIL, f"mean off-diagonal cosine {cos:.4f} > {self.threshold}"
        return Verdict.PASS, None

    def _measured(self, target: torch.Tensor) -> Any:
        return self._mean_offdiag_cos(target)
