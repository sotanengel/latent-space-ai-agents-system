"""HiddenStateTap: observer that captures `hidden_states` from a HF model.

Covers FR-LG-001 (shape), FR-LG-002 (dtype), FR-LG-004 (NaN/Inf flagging),
FR-LG-007 (deterministic reproducibility) via the metadata it attaches.

Design: pure observer. We do not patch the model graph; we intercept the
returned `ModelOutput.hidden_states` tuple inside the `with` block by
wrapping `forward`. The captured tensors are detached + cloned to CPU so
nothing the caller does afterwards can mutate them (and vice versa).
"""

from __future__ import annotations

import hashlib
import math
from types import TracebackType
from typing import Any

import torch

from latent_agents.schemas import (
    Context,
    Governance,
    LatentRepresentation,
    NumericalProperties,
    Source,
    TensorMeta,
)

_DTYPE_TO_STR: dict[torch.dtype, str] = {
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.int8: "int8",
}


class HiddenStateTap:
    """Capture hidden_states for the duration of a `with` block."""

    def __init__(
        self,
        model: Any,
        *,
        layers: str | list[int] = "all",
        agent_id: str = "anonymous",
    ) -> None:
        self._model = model
        self._layers = layers
        self._agent_id = agent_id
        self._original_forward: Any = None
        self.snapshots: list[LatentRepresentation] = []
        self.tensors: list[torch.Tensor] = []

    # --- context manager -------------------------------------------------

    def __enter__(self) -> HiddenStateTap:
        if hasattr(self._model, "forward"):
            self._original_forward = self._model.forward
            tap = self

            def _wrapped(*args: Any, **kwargs: Any) -> Any:
                output = tap._original_forward(*args, **kwargs)
                hs = getattr(output, "hidden_states", None)
                if hs is not None and len(hs) > 0:
                    tap._on_hidden_states(hs)
                return output

            self._model.forward = _wrapped
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._original_forward is not None:
            self._model.forward = self._original_forward
            self._original_forward = None

    # --- core capture ----------------------------------------------------

    def _on_hidden_states(self, hidden_states: tuple[torch.Tensor, ...]) -> None:
        selected = self._select(hidden_states)
        cfg = getattr(self._model, "config", None)
        for layer_idx, h in selected:
            t = h.detach().to("cpu").clone()
            self.tensors.append(t)
            self.snapshots.append(self._to_latent_representation(t, layer_idx=layer_idx, cfg=cfg))

    def _select(self, hidden_states: tuple[torch.Tensor, ...]) -> list[tuple[int, torch.Tensor]]:
        if self._layers == "all":
            return list(enumerate(hidden_states))
        if isinstance(self._layers, list):
            return [(i, hidden_states[i]) for i in self._layers]
        raise TypeError(f"layers must be 'all' or list[int], got {type(self._layers)}")

    def _to_latent_representation(
        self, tensor: torch.Tensor, *, layer_idx: int, cfg: Any
    ) -> LatentRepresentation:
        dtype_str = _DTYPE_TO_STR.get(tensor.dtype, "fp32")
        has_nan = bool(torch.isnan(tensor).any().item())
        has_inf = bool(torch.isinf(tensor).any().item())

        finite = tensor[torch.isfinite(tensor)]
        if finite.numel() == 0:
            l2 = float("nan")
            mean = float("nan")
            std = float("nan")
            sparsity = 0.0
        else:
            l2 = float(torch.linalg.vector_norm(finite).item())
            mean = float(finite.mean().item())
            std = float(finite.std(unbiased=False).item())
            sparsity = float((tensor == 0).float().mean().item())

        # Stable content hash for de-duplication / provenance.
        content_hash = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()[:16]

        model_family = type(getattr(cfg, "__class__", object)).__name__ if cfg else "unknown"
        if cfg is not None:
            model_family = cfg.__class__.__name__.replace("Config", "")

        return LatentRepresentation(
            tensor=TensorMeta(
                shape=tuple(tensor.shape),
                dtype=dtype_str,
                storage_ref=f"memory://{content_hash}",
            ),
            source=Source(
                agent_id=self._agent_id,
                model_family=model_family or "unknown",
                model_version=str(getattr(cfg, "_name_or_path", "unknown")),
                layer_index=layer_idx,
            ),
            context=Context(input_hash=f"sha256:{content_hash}"),
            numerical_properties=NumericalProperties(
                l2_norm=l2 if not math.isnan(l2) else 0.0,
                mean=mean if not math.isnan(mean) else 0.0,
                std=std if not math.isnan(std) else 0.0,
                sparsity=sparsity,
                has_nan=has_nan,
                has_inf=has_inf,
            ),
            governance=Governance(retention_policy="TTL:1h"),
        )
