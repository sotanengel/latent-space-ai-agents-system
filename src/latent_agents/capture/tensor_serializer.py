"""Tensor serializer using `safetensors`.

Why safetensors over `torch.save`:
- no pickle (supply-chain safety per CLAUDE.md / Takumi Guard intent)
- header-only metadata makes random-access loading cheap
- guarantees bit-exact roundtrip (FR-KV-001) when dtype/shape are preserved
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
from safetensors import safe_open
from safetensors.torch import save_file


class TensorSerializer:
    """Persist `dict[str, torch.Tensor]` to disk with optional metadata."""

    def save(
        self,
        tensors: dict[str, torch.Tensor],
        path: str | Path,
        *,
        metadata: dict[str, str] | None = None,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # safetensors requires contiguous tensors on CPU
        prepared = {k: v.detach().cpu().contiguous() for k, v in tensors.items()}
        save_file(prepared, str(path), metadata=metadata or {})
        return path

    def load(self, path: str | Path) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        with safe_open(str(path), framework="pt") as f:
            for k in f.keys():  # noqa: SIM118 - safetensors API exposes .keys() method
                out[k] = cast(torch.Tensor, f.get_tensor(k))
        return out

    def load_metadata(self, path: str | Path) -> dict[str, str]:
        with safe_open(str(path), framework="pt") as f:
            md = f.metadata()
        return dict(md) if md else {}

    @staticmethod
    def bit_equal(
        a: dict[str, torch.Tensor],
        b: dict[str, torch.Tensor],
    ) -> bool:
        if set(a.keys()) != set(b.keys()):
            return False
        for k, av in a.items():
            bv = b[k]
            if av.dtype != bv.dtype or av.shape != bv.shape:
                return False
            if not torch.equal(av, bv):
                return False
        return True
