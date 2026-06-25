"""LatentRepresentation schema (§6.1).

The `LatentRepresentation` model intentionally does NOT carry the tensor
itself; it carries shape / dtype / numerical properties plus a `storage_ref`
pointing at the actual bytes (safetensors on disk, in-memory blob, etc.).
This keeps schema-level operations cheap and lets us hash content for
content-addressable IDs.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DType = Literal["fp16", "bf16", "fp32", "int8", "int4", "fp8"]


class TensorMeta(BaseModel):
    model_config = ConfigDict(frozen=True)

    shape: tuple[int, ...]
    dtype: DType
    storage_ref: str

    @model_validator(mode="after")
    def _validate_shape(self) -> TensorMeta:
        if len(self.shape) != 3:
            raise ValueError(
                f"hidden_states tensor shape must be [batch, seq_len, hidden_dim], got {self.shape}"
            )
        if any(d <= 0 for d in self.shape):
            raise ValueError(f"all shape dims must be > 0, got {self.shape}")
        return self


class Source(BaseModel):
    model_config = ConfigDict(frozen=True)

    agent_id: str
    model_family: str
    model_version: str
    layer_index: int
    latent_step: int = 0


class Context(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_hash: str
    parent_latents: list[str] = Field(default_factory=list)
    realignment_applied: bool = False
    realignment_matrix_id: str | None = None


class NumericalProperties(BaseModel):
    model_config = ConfigDict(frozen=True)

    l2_norm: float
    mean: float
    std: float
    sparsity: float = Field(ge=0.0, le=1.0)
    has_nan: bool
    has_inf: bool
    skewness: float | None = None


class RetentionPolicy(StrEnum):
    PERSISTENT = "persistent"
    EPHEMERAL = "ephemeral"


class Governance(BaseModel):
    model_config = ConfigDict(frozen=True)

    trust_boundary: str | None = None
    signature: str | None = None
    retention_policy: str = "TTL:1h"


class LatentRepresentation(BaseModel):
    """Metadata-only handle to a latent representation tensor."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default="")
    tensor: TensorMeta
    source: Source
    context: Context
    numerical_properties: NumericalProperties
    timestamps: dict[str, datetime] = Field(
        default_factory=lambda: {"created_at": datetime.now(UTC)}
    )
    governance: Governance

    @model_validator(mode="after")
    def _derive_id(self) -> LatentRepresentation:
        if not self.id:
            payload = self.model_dump_json(exclude={"id", "timestamps"})
            digest = hashlib.sha256(payload.encode()).hexdigest()
            object.__setattr__(self, "id", f"sha256:{digest}")
        return self
