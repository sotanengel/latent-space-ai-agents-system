"""CrossModalValidator (FR-MM-001..005).

Pieces:
- Modality + ModalityTypedTensor (FR-MM-004): type-tagged tensors prevent
  silent cross-modal contamination in shared working memory.
- VisionLanguageProjector (FR-MM-002): linear projection from a vision
  encoder's hidden dim to an LLM's input embedding dim.
- MultiModalKVCacheLayout (FR-MM-003): records modality-spans inside one
  KV cache so downstream code can verify text-vs-image token boundaries.
- CrossModalAlignmentVerifier (FR-MM-001/005): mean per-row cosine between
  two modality feature blocks.
- CrossModalValidator orchestrates the bundle for a single
  vision-vs-language check.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch
from torch import nn

from latent_agents.schemas import Verdict
from latent_agents.verifier import BaseVerifier, register


class Modality(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass(frozen=True)
class ModalityTypedTensor:
    modality: Modality
    data: torch.Tensor

    def assert_same_modality(self, other: ModalityTypedTensor) -> None:
        if self.modality != other.modality:
            raise ValueError(f"modality mismatch: {self.modality.value} vs {other.modality.value}")


class VisionLanguageProjector:
    """Linear `[vision_dim] -> [language_dim]` projection (FR-MM-002)."""

    def __init__(self, *, vision_dim: int, language_dim: int) -> None:
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self._proj = nn.Linear(vision_dim, language_dim, bias=False)

    def project(self, vision_features: torch.Tensor) -> torch.Tensor:
        if vision_features.shape[-1] != self.vision_dim:
            raise ValueError(
                f"expected last-dim {self.vision_dim}, got {vision_features.shape[-1]}"
            )
        with torch.no_grad():
            out: torch.Tensor = self._proj(vision_features)
        return out


@dataclass(frozen=True)
class MultiModalKVCacheLayout:
    """Records per-modality token spans inside a unified KV cache (FR-MM-003)."""

    modalities: list[Modality]
    spans: list[tuple[int, int]]

    def __post_init__(self) -> None:
        if len(self.modalities) != len(self.spans):
            raise ValueError("modalities and spans must align")
        prev_end = 0
        for start, end in self.spans:
            if start < prev_end:
                raise ValueError(f"span {(start, end)} overlaps previous (ended at {prev_end})")
            if end <= start:
                raise ValueError(f"span end {end} must be > start {start}")
            prev_end = end

    @property
    def length(self) -> int:
        return self.spans[-1][1] if self.spans else 0

    def modality_at(self, token_index: int) -> Modality:
        for m, (start, end) in zip(self.modalities, self.spans, strict=True):
            if start <= token_index < end:
                return m
        raise IndexError(f"token_index {token_index} out of range")


@register
class CrossModalAlignmentVerifier(BaseVerifier[tuple[torch.Tensor, torch.Tensor]]):
    """FR-MM-001/005: cross-modal mean cosine similarity must be >= min_cosine."""

    test_id = "FR-MM-001"

    def __init__(self, min_cosine: float = 0.5) -> None:
        super().__init__()
        self.threshold = min_cosine

    @staticmethod
    def _mean_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        a_n = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        b_n = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        return float((a_n * b_n).sum(dim=-1).mean().item())

    def _evaluate(self, target: tuple[torch.Tensor, torch.Tensor]) -> tuple[Verdict, str | None]:
        cos = self._mean_cosine(*target)
        if cos < float(self.threshold):
            return Verdict.FAIL, f"mean cosine {cos:.4f} < {self.threshold}"
        return Verdict.PASS, None

    def _measured(self, target: tuple[torch.Tensor, torch.Tensor]) -> float:
        return self._mean_cosine(*target)


class CrossModalValidator:
    """Run the cross-modal bundle for a single (vision, language) pair."""

    def __init__(
        self,
        *,
        projector: VisionLanguageProjector | None = None,
        min_cosine: float = 0.5,
    ) -> None:
        self._projector = projector
        self._aligner = CrossModalAlignmentVerifier(min_cosine=min_cosine)

    def validate(
        self,
        *,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
    ) -> dict[str, Any]:
        report: dict[str, Any] = {}
        if self._projector is not None:
            projected = self._projector.project(vision_features)
            report["projection_shape"] = tuple(projected.shape)
            target = projected
        else:
            target = vision_features
        result = self._aligner.verify((target, language_features))
        report["alignment"] = result.verdict
        report["cosine"] = result.measured_value
        return report
