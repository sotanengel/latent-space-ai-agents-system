"""KVCacheTap: capture and validate `past_key_values` structure.

Covers FR-KV-002 (shape), FR-KV-003 (layer count match), FR-KV-004
(num_heads / head_dim mismatch detection), FR-KV-005 (position encoding
type carried alongside), FR-KV-006 (GQA / MQA num_kv_heads).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class LayerKV:
    """One layer's Key/Value tensors with the metadata needed to validate them."""

    layer_index: int
    key: torch.Tensor
    value: torch.Tensor
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    position_encoding: str

    @property
    def is_gqa(self) -> bool:
        return self.num_kv_heads != self.num_attention_heads


def _normalise(
    past_key_values: Any,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Accept legacy tuple-of-tuples and transformers >=4.43 `DynamicCache`."""
    # DynamicCache exposes .key_cache / .value_cache parallel lists.
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        keys = list(past_key_values.key_cache)
        vals = list(past_key_values.value_cache)
        if len(keys) != len(vals):
            raise ValueError("DynamicCache key/value layer count mismatch")
        return list(zip(keys, vals, strict=True))
    # Legacy: tuple[tuple[K, V], ...]
    return [(k, v) for k, v in past_key_values]


class KVCacheTap:
    """Validates a KV cache against `model.config` and wraps it in `LayerKV`s."""

    def __init__(self, config: Any) -> None:
        self.config = config

    # context-manager API is provided for symmetry with HiddenStateTap;
    # for KV the user typically calls `capture()` explicitly.
    def __enter__(self) -> KVCacheTap:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def capture(self, past_key_values: Any) -> list[LayerKV]:
        layers = _normalise(past_key_values)
        cfg = self.config

        expected_layers = int(getattr(cfg, "num_hidden_layers", len(layers)))
        if len(layers) != expected_layers:
            raise ValueError(
                f"layer count mismatch: got {len(layers)} cached layers, "
                f"model.config.num_hidden_layers = {expected_layers}"
            )

        num_attention_heads = int(getattr(cfg, "num_attention_heads", 0))
        num_kv_heads = int(
            getattr(cfg, "num_key_value_heads", num_attention_heads) or num_attention_heads
        )
        head_dim = int(
            getattr(cfg, "head_dim", 0)
            or (int(getattr(cfg, "hidden_size", 0)) // max(num_attention_heads, 1))
        )
        position_encoding = str(getattr(cfg, "position_embedding_type", "rope"))

        captured: list[LayerKV] = []
        for idx, (k, v) in enumerate(layers):
            self._validate_layer(k, v, num_kv_heads=num_kv_heads, head_dim=head_dim, idx=idx)
            captured.append(
                LayerKV(
                    layer_index=idx,
                    key=k.detach().cpu().clone(),
                    value=v.detach().cpu().clone(),
                    num_attention_heads=num_attention_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    position_encoding=position_encoding,
                )
            )
        return captured

    @staticmethod
    def _validate_layer(
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        num_kv_heads: int,
        head_dim: int,
        idx: int,
    ) -> None:
        if k.shape != v.shape:
            raise ValueError(f"layer {idx}: key/value shape mismatch {k.shape} vs {v.shape}")
        if k.ndim != 4:
            raise ValueError(
                f"layer {idx}: expected [batch, num_heads, seq_len, head_dim], got {tuple(k.shape)}"
            )
        if k.shape[1] != num_kv_heads:
            raise ValueError(
                f"layer {idx}: num_heads mismatch — tensor has {k.shape[1]}, "
                f"config expects {num_kv_heads}"
            )
        if k.shape[3] != head_dim:
            raise ValueError(
                f"layer {idx}: head_dim mismatch — tensor has {k.shape[3]}, "
                f"config expects {head_dim}"
            )
