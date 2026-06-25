"""Tests for KVCacheTap (FR-KV-002..006)."""

from __future__ import annotations

import pytest
import torch

from latent_agents.capture import KVCacheTap, LayerKV


class _StubConfig:
    def __init__(
        self,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 4,
        num_key_value_heads: int | None = None,
        head_dim: int = 8,
        position_embedding_type: str = "rope",
    ) -> None:
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is not None:
            self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_size = num_attention_heads * head_dim
        self.position_embedding_type = position_embedding_type


def _legacy_kv(
    layers: int = 2, batch: int = 1, heads: int = 4, seq: int = 3, head_dim: int = 8
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    return tuple(
        (torch.randn(batch, heads, seq, head_dim), torch.randn(batch, heads, seq, head_dim))
        for _ in range(layers)
    )


class TestKVCacheTap:
    def test_capture_legacy_tuple_form(self) -> None:
        cfg = _StubConfig(num_hidden_layers=2)
        tap = KVCacheTap(cfg)
        kv = _legacy_kv(layers=2)
        layers = tap.capture(kv)
        assert len(layers) == 2
        assert all(isinstance(layer, LayerKV) for layer in layers)
        assert layers[0].key.shape == (1, 4, 3, 8)

    def test_layer_count_must_match_config(self) -> None:
        """FR-KV-003: send-side and receive-side layer count must match."""
        cfg = _StubConfig(num_hidden_layers=2)
        tap = KVCacheTap(cfg)
        wrong = _legacy_kv(layers=3)
        with pytest.raises(ValueError, match="layer count mismatch"):
            tap.capture(wrong)

    def test_head_dim_mismatch_raises(self) -> None:
        """FR-KV-004: num_heads / head_dim mismatch should be detected eagerly."""
        cfg = _StubConfig(num_hidden_layers=1, num_attention_heads=4, head_dim=8)
        tap = KVCacheTap(cfg)
        bad = ((torch.randn(1, 4, 3, 7), torch.randn(1, 4, 3, 7)),)  # head_dim 7 not 8
        with pytest.raises(ValueError, match="head_dim"):
            tap.capture(bad)

    def test_gqa_num_kv_heads_recorded(self) -> None:
        """FR-KV-006: GQA / MQA structure (num_kv_heads != num_heads) preserved."""
        cfg = _StubConfig(
            num_hidden_layers=1,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=8,
        )
        tap = KVCacheTap(cfg)
        kv = ((torch.randn(1, 2, 3, 8), torch.randn(1, 2, 3, 8)),)  # kv heads = 2
        layers = tap.capture(kv)
        assert layers[0].num_kv_heads == 2
        assert layers[0].num_attention_heads == 8

    def test_position_encoding_type_recorded(self) -> None:
        """FR-KV-005: position encoding scheme should travel with the cache."""
        cfg = _StubConfig(position_embedding_type="alibi")
        tap = KVCacheTap(cfg)
        kv = _legacy_kv(layers=2)
        layers = tap.capture(kv)
        assert all(layer.position_encoding == "alibi" for layer in layers)
