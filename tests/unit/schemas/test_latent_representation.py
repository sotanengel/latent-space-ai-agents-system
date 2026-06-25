"""Tests for LatentRepresentation schema (§6.1)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from latent_agents.schemas import (
    Context,
    Governance,
    LatentRepresentation,
    NumericalProperties,
    Source,
    TensorMeta,
)


def _build(**overrides: object) -> LatentRepresentation:
    base: dict[str, object] = {
        "tensor": TensorMeta(shape=(1, 4, 8), dtype="fp16", storage_ref="mem://test"),
        "source": Source(
            agent_id="agent-a",
            model_family="Qwen",
            model_version="14B",
            layer_index=11,
            latent_step=0,
        ),
        "context": Context(input_hash="sha256:deadbeef"),
        "numerical_properties": NumericalProperties(
            l2_norm=1.0, mean=0.0, std=0.5, sparsity=0.1, has_nan=False, has_inf=False
        ),
        "governance": Governance(trust_boundary=None, signature=None, retention_policy="TTL:1h"),
    }
    base.update(overrides)
    return LatentRepresentation(**base)  # type: ignore[arg-type]


class TestLatentRepresentation:
    def test_id_derived_from_content_when_omitted(self) -> None:
        l1 = _build()
        l2 = _build()
        assert l1.id == l2.id, "same content must hash to same id"
        assert l1.id.startswith("sha256:")

    def test_id_changes_when_content_changes(self) -> None:
        l1 = _build()
        l2 = _build(tensor=TensorMeta(shape=(2, 4, 8), dtype="fp16", storage_ref="mem://test"))
        assert l1.id != l2.id

    def test_tensor_meta_dtype_must_be_known(self) -> None:
        with pytest.raises(ValidationError):
            TensorMeta(shape=(1, 2, 3), dtype="binary42", storage_ref="x")  # type: ignore[arg-type]

    def test_shape_must_be_three_dim(self) -> None:
        with pytest.raises(ValidationError):
            TensorMeta(shape=(1, 2), dtype="fp16", storage_ref="x")

    def test_numerical_properties_nan_inf_flags(self) -> None:
        l = _build(  # noqa: E741
            numerical_properties=NumericalProperties(
                l2_norm=float("nan"), mean=0, std=1, sparsity=0, has_nan=True, has_inf=False
            )
        )
        assert l.numerical_properties.has_nan is True

    def test_serialize_roundtrip(self) -> None:
        l = _build()  # noqa: E741
        data = l.model_dump_json()
        restored = LatentRepresentation.model_validate_json(data)
        assert restored.id == l.id
        assert restored.tensor.shape == (1, 4, 8)

    def test_parent_latents_default_empty(self) -> None:
        l = _build()  # noqa: E741
        assert l.context.parent_latents == []
