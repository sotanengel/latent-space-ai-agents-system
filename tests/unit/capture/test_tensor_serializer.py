"""Tests for TensorSerializer (FR-KV-001 bit-exact roundtrip, FR-KV-007 precision)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from latent_agents.capture import TensorSerializer


@pytest.fixture
def tmp_path_safetensor(tmp_path: Path) -> Path:
    return tmp_path / "k.safetensors"


class TestTensorSerializer:
    def test_roundtrip_bit_exact_fp32(self, tmp_path_safetensor: Path) -> None:
        ser = TensorSerializer()
        tensors = {"k_0": torch.randn(2, 4, 8), "v_0": torch.randn(2, 4, 8)}
        ser.save(tensors, tmp_path_safetensor)
        loaded = ser.load(tmp_path_safetensor)
        assert ser.bit_equal(tensors, loaded)  # FR-KV-001

    def test_roundtrip_bit_exact_fp16(self, tmp_path_safetensor: Path) -> None:
        ser = TensorSerializer()
        tensors = {"k_0": torch.randn(2, 4, 8).half()}
        ser.save(tensors, tmp_path_safetensor)
        loaded = ser.load(tmp_path_safetensor)
        assert ser.bit_equal(tensors, loaded)

    def test_metadata_is_preserved(self, tmp_path_safetensor: Path) -> None:
        ser = TensorSerializer()
        tensors = {"k_0": torch.zeros(1, 1, 1)}
        meta = {"layer": "0", "agent": "a"}
        ser.save(tensors, tmp_path_safetensor, metadata=meta)
        meta_back = ser.load_metadata(tmp_path_safetensor)
        assert meta_back == meta

    def test_bit_equal_detects_mismatch(self) -> None:
        ser = TensorSerializer()
        a = {"k": torch.tensor([1.0, 2.0])}
        b = {"k": torch.tensor([1.0, 2.001])}
        assert not ser.bit_equal(a, b)

    def test_bit_equal_detects_dtype_mismatch(self) -> None:
        ser = TensorSerializer()
        a = {"k": torch.tensor([1.0, 2.0], dtype=torch.float32)}
        b = {"k": torch.tensor([1.0, 2.0], dtype=torch.float16)}
        assert not ser.bit_equal(a, b)

    def test_bit_equal_detects_missing_key(self) -> None:
        ser = TensorSerializer()
        a = {"k": torch.zeros(2)}
        b = {"k": torch.zeros(2), "v": torch.zeros(2)}
        assert not ser.bit_equal(a, b)
