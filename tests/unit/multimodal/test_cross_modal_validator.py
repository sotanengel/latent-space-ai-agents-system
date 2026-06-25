"""Tests for CrossModalValidator (FR-MM-001..005)."""

from __future__ import annotations

import pytest
import torch

from latent_agents.multimodal import (
    CrossModalAlignmentVerifier,
    CrossModalValidator,
    Modality,
    ModalityTypedTensor,
    MultiModalKVCacheLayout,
    VisionLanguageProjector,
)
from latent_agents.schemas import Verdict


class TestModalityTypedTensor:
    def test_modality_is_recorded(self) -> None:
        t = ModalityTypedTensor(modality=Modality.TEXT, data=torch.zeros(4, 8))
        assert t.modality == Modality.TEXT

    def test_rejects_mismatched_assignment(self) -> None:
        a = ModalityTypedTensor(modality=Modality.TEXT, data=torch.zeros(4, 8))
        b = ModalityTypedTensor(modality=Modality.IMAGE, data=torch.zeros(4, 8))
        with pytest.raises(ValueError, match="modality mismatch"):
            a.assert_same_modality(b)


class TestVisionLanguageProjector:
    def test_projects_vision_to_language_dim(self) -> None:
        proj = VisionLanguageProjector(vision_dim=512, language_dim=128)
        v = torch.randn(16, 512)
        out = proj.project(v)
        assert out.shape == (16, 128)


class TestMultiModalKVCacheLayout:
    def test_layout_records_modal_boundaries(self) -> None:
        layout = MultiModalKVCacheLayout(
            modalities=[Modality.TEXT, Modality.IMAGE, Modality.TEXT],
            spans=[(0, 4), (4, 20), (20, 25)],
        )
        assert layout.length == 25
        assert layout.modality_at(10) == Modality.IMAGE

    def test_overlapping_spans_rejected(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            MultiModalKVCacheLayout(
                modalities=[Modality.TEXT, Modality.IMAGE],
                spans=[(0, 10), (5, 15)],
            )


class TestCrossModalAlignmentVerifier:
    def test_aligned_modalities_pass(self) -> None:
        x = torch.randn(64, 16)
        v = CrossModalAlignmentVerifier(min_cosine=0.5)
        r = v.verify((x, x))
        assert r.verdict == Verdict.PASS

    def test_orthogonal_modalities_fail(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(64, 16)
        y = torch.randn(64, 16)
        v = CrossModalAlignmentVerifier(min_cosine=0.5)
        r = v.verify((x, y))
        assert r.verdict == Verdict.FAIL


class TestCrossModalValidator:
    def test_runs_full_bundle_without_projector(self) -> None:
        val = CrossModalValidator(projector=None, min_cosine=0.5)
        x = torch.randn(16, 8)
        report = val.validate(vision_features=x, language_features=x)
        assert report["alignment"] == Verdict.PASS

    def test_projection_shape_recorded_when_projector_supplied(self) -> None:
        val = CrossModalValidator(
            projector=VisionLanguageProjector(vision_dim=8, language_dim=4),
            min_cosine=0.0,  # accept any alignment so we only check shape plumbing
        )
        x = torch.randn(16, 8)
        y = torch.randn(16, 4)
        report = val.validate(vision_features=x, language_features=y)
        assert report["projection_shape"] == (16, 4)
