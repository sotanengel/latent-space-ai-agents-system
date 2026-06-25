"""Realignment matrix validator (Phase 2).

Implements:
- ProcrustesAligner: training-free orthogonal Wa from two paired matrices.
- ConditionNumberVerifier (FR-RA-002), SingularValueVerifier (FR-RA-003),
  OrthogonalityVerifier (FR-RA-004), RoundTripVerifier (FR-RA-008).
- RealignmentValidator: convenience that runs the bundle.

The aligner is *training-free* (FR-RA-006) and deterministic (closed-form
SVD), which keeps the realignment cache-once-reuse design (FR-RA-009) safe.
"""

from __future__ import annotations

import torch

from latent_agents.schemas import Verdict, VerificationResult
from latent_agents.verifier import BaseVerifier, register


class ProcrustesAligner:
    """Closed-form orthogonal Procrustes: find orthogonal W minimising ||X W - Y||."""

    @staticmethod
    def fit(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if source.shape != target.shape:
            raise ValueError(
                f"Procrustes requires matching shapes, got {tuple(source.shape)} vs "
                f"{tuple(target.shape)}"
            )
        m = source.t().to(torch.float64) @ target.to(torch.float64)
        u, _, vt = torch.linalg.svd(m, full_matrices=False)
        wa: torch.Tensor = u @ vt
        return wa.to(source.dtype)


@register
class ConditionNumberVerifier(BaseVerifier[torch.Tensor]):
    """FR-RA-002: condition number kappa(Wa) <= max_kappa."""

    test_id = "FR-RA-002"

    def __init__(self, max_kappa: float = 1e4) -> None:
        super().__init__()
        self.threshold = max_kappa

    @staticmethod
    def _kappa(wa: torch.Tensor) -> float:
        s = torch.linalg.svdvals(wa.to(torch.float64))
        return float((s.max() / s.min().clamp_min(1e-30)).item())

    def _evaluate(self, target: torch.Tensor) -> tuple[Verdict, str | None]:
        k = self._kappa(target)
        if k > float(self.threshold):
            return Verdict.FAIL, f"kappa(Wa)={k:.3e} > {self.threshold:.0e}"
        return Verdict.PASS, None

    def _measured(self, target: torch.Tensor) -> float:
        return self._kappa(target)


@register
class SingularValueVerifier(BaseVerifier[torch.Tensor]):
    """FR-RA-003: minimum singular value of Wa must be >= min_singular."""

    test_id = "FR-RA-003"

    def __init__(self, min_singular: float = 1e-6) -> None:
        super().__init__()
        self.threshold = min_singular

    @staticmethod
    def _min_sv(wa: torch.Tensor) -> float:
        return float(torch.linalg.svdvals(wa.to(torch.float64)).min().item())

    def _evaluate(self, target: torch.Tensor) -> tuple[Verdict, str | None]:
        m = self._min_sv(target)
        if m < float(self.threshold):
            return Verdict.FAIL, f"min_sv={m:.3e} < {self.threshold:.0e}"
        return Verdict.PASS, None

    def _measured(self, target: torch.Tensor) -> float:
        return self._min_sv(target)


@register
class OrthogonalityVerifier(BaseVerifier[torch.Tensor]):
    """FR-RA-004: ||Wa^T Wa - I||_F must be < tol."""

    test_id = "FR-RA-004"

    def __init__(self, tol: float = 1e-4) -> None:
        super().__init__()
        self.threshold = tol

    @staticmethod
    def _frob_dev(wa: torch.Tensor) -> float:
        w64 = wa.to(torch.float64)
        n = w64.shape[0]
        eye = torch.eye(n, dtype=torch.float64)
        return float((w64.t() @ w64 - eye).norm().item())

    def _evaluate(self, target: torch.Tensor) -> tuple[Verdict, str | None]:
        f = self._frob_dev(target)
        if f >= float(self.threshold):
            return Verdict.FAIL, f"||W^T W - I||_F = {f:.3e} >= {self.threshold:.0e}"
        return Verdict.PASS, None

    def _measured(self, target: torch.Tensor) -> float:
        return self._frob_dev(target)


@register
class RoundTripVerifier(BaseVerifier[tuple[torch.Tensor, torch.Tensor]]):
    """FR-RA-008: round-trip error ||X - Wa^{-1} Wa X|| / ||X|| must be < tol."""

    test_id = "FR-RA-008"

    def __init__(self, tol: float = 1e-3) -> None:
        super().__init__()
        self.threshold = tol

    def _residual(self, target: tuple[torch.Tensor, torch.Tensor]) -> float:
        wa, x = target
        # Use the input dtype (typically float32) so numerical error from
        # ill-conditioned Wa is exposed instead of papered over by float64.
        wa_inv = torch.linalg.inv(wa)
        rt = (x @ wa) @ wa_inv
        denom = x.norm().clamp_min(1e-30)
        return float(((rt - x).norm() / denom).item())

    def _evaluate(self, target: tuple[torch.Tensor, torch.Tensor]) -> tuple[Verdict, str | None]:
        try:
            r = self._residual(target)
        except RuntimeError as exc:
            # torch.linalg.inv raises torch._C._LinAlgError (subclass of RuntimeError)
            # for singular inputs.
            return Verdict.FAIL, f"Wa is singular: {exc}"
        if r >= float(self.threshold):
            return Verdict.FAIL, f"roundtrip residual {r:.3e} >= {self.threshold:.0e}"
        return Verdict.PASS, None

    def _measured(self, target: tuple[torch.Tensor, torch.Tensor]) -> float:
        try:
            return self._residual(target)
        except RuntimeError:
            return float("inf")


class RealignmentValidator:
    """Run the standard Wa verifier suite and collect results."""

    def __init__(
        self,
        *,
        max_kappa: float = 1e4,
        min_singular: float = 1e-6,
        orth_tol: float = 1e-4,
    ) -> None:
        self._verifiers: list[BaseVerifier[torch.Tensor]] = [
            _ShapeVerifier(),
            ConditionNumberVerifier(max_kappa=max_kappa),
            SingularValueVerifier(min_singular=min_singular),
            OrthogonalityVerifier(tol=orth_tol),
        ]

    def validate(self, wa: torch.Tensor) -> list[VerificationResult]:
        return [v.verify(wa) for v in self._verifiers]


@register
class _ShapeVerifier(BaseVerifier[torch.Tensor]):
    """FR-RA-001: Wa shape must be [hidden_dim, hidden_dim]."""

    test_id = "FR-RA-001"

    def _evaluate(self, target: torch.Tensor) -> tuple[Verdict, str | None]:
        if target.ndim != 2 or target.shape[0] != target.shape[1]:
            return Verdict.FAIL, f"expected square 2D matrix, got {tuple(target.shape)}"
        return Verdict.PASS, None

    def _measured(self, target: torch.Tensor) -> tuple[int, ...]:
        return tuple(target.shape)
