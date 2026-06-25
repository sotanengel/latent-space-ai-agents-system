"""Cross-model latent alignment (Phase 2).

- LinearProjectionAligner: least-squares projection between models with
  different hidden_dim (FR-CM-004).
- CKAVerifier: linear Centered Kernel Alignment for cross-model latent
  similarity (FR-CM-007).
- CrossModelAligner: dispatches to Procrustes when dims match, projection
  otherwise.
"""

from __future__ import annotations

import torch

from latent_agents.analysis.realignment_validator import ProcrustesAligner
from latent_agents.schemas import Verdict
from latent_agents.verifier import BaseVerifier, register


class LinearProjectionAligner:
    """Least-squares projection W (shape `[d_src, d_tgt]`) so X @ W ~= Y."""

    @staticmethod
    def fit(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if source.shape[0] != target.shape[0]:
            raise ValueError("source/target must share batch dim")
        s64 = source.to(torch.float64)
        t64 = target.to(torch.float64)
        sol: torch.Tensor = torch.linalg.lstsq(s64, t64).solution
        return sol.to(source.dtype)


@register
class CKAVerifier(BaseVerifier[tuple[torch.Tensor, torch.Tensor]]):
    """FR-CM-007: linear CKA between two feature matrices must be >= min_cka."""

    test_id = "FR-CM-007"

    def __init__(self, min_cka: float = 0.7) -> None:
        super().__init__()
        self.threshold = min_cka

    @staticmethod
    def _cka(x: torch.Tensor, y: torch.Tensor) -> float:
        xc = x.to(torch.float64) - x.to(torch.float64).mean(0, keepdim=True)
        yc = y.to(torch.float64) - y.to(torch.float64).mean(0, keepdim=True)
        num = (xc.t() @ yc).norm() ** 2
        denom = (xc.t() @ xc).norm() * (yc.t() @ yc).norm()
        return float((num / denom.clamp_min(1e-30)).item())

    def _evaluate(self, target: tuple[torch.Tensor, torch.Tensor]) -> tuple[Verdict, str | None]:
        x, y = target
        cka = self._cka(x, y)
        if cka < float(self.threshold):
            return Verdict.FAIL, f"CKA={cka:.4f} < {self.threshold}"
        return Verdict.PASS, None

    def _measured(self, target: tuple[torch.Tensor, torch.Tensor]) -> float:
        return self._cka(*target)


class CrossModelAligner:
    """Choose Procrustes when dims match, projection otherwise."""

    def align(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if source.shape[1] == target.shape[1]:
            return ProcrustesAligner().fit(source, target)
        return LinearProjectionAligner().fit(source, target)
