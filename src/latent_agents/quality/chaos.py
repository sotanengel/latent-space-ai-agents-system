"""Failure injection toolkit (FR-TA-005).

Primitives:
- `flip_bits`: corrupt a tensor's payload at the byte level.
- `delay_call`: wrap a callable so it sleeps before executing.
- `ChaosInjector`: probabilistic injector with named failure modes.

These are intentionally small so chaos tests can compose them without
pulling in a full chaos-engineering framework.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from enum import StrEnum
from typing import TypeVar

import torch

R = TypeVar("R")


class FailureMode(StrEnum):
    AGENT_KILL = "agent_kill"
    COMMS_DELAY = "comms_delay"
    TENSOR_CORRUPT = "tensor_corrupt"
    OOM = "oom"
    CLOCK_SKEW = "clock_skew"
    HANG = "hang"
    NETWORK_PARTITION = "network_partition"
    SCHEMA_DRIFT = "schema_drift"
    KEY_ROTATION_RACE = "key_rotation_race"
    REPLAY_ATTACK = "replay_attack"


def flip_bits(tensor: torch.Tensor, *, n: int = 1, seed: int | None = None) -> torch.Tensor:
    """Return a copy of `tensor` with `n` random bytes XOR'd by 0xFF."""
    rng = random.Random(seed)
    flat = tensor.detach().clone().cpu().contiguous().view(torch.uint8)
    out = flat.clone()
    size = out.numel()
    if size == 0:
        return tensor
    indices = [rng.randrange(size) for _ in range(n)]
    for i in indices:
        out[i] = out[i] ^ 0xFF
    return out.view(tensor.dtype).view(tensor.shape)


def delay_call(fn: Callable[..., R], *, delay_seconds: float) -> Callable[..., R]:
    def wrapped(*args: object, **kwargs: object) -> R:
        time.sleep(delay_seconds)
        return fn(*args, **kwargs)

    return wrapped


class ChaosInjector:
    """Probabilistic injector for the failure modes the chaos tests exercise."""

    def __init__(self, *, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._active: dict[FailureMode, float] = {}

    def activate(self, mode: FailureMode, *, probability: float = 1.0) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability must be in [0,1], got {probability}")
        self._active[mode] = probability

    def maybe_fail(self) -> None:
        if (
            FailureMode.AGENT_KILL in self._active
            and self._rng.random() < self._active[FailureMode.AGENT_KILL]
        ):
            raise RuntimeError("injected agent kill")
        if FailureMode.HANG in self._active and self._rng.random() < self._active[FailureMode.HANG]:
            # Bounded short hang so test suites don't lock up.
            time.sleep(0.001)

    def maybe_corrupt(self, tensor: torch.Tensor) -> torch.Tensor:
        if (
            FailureMode.TENSOR_CORRUPT in self._active
            and self._rng.random() < self._active[FailureMode.TENSOR_CORRUPT]
        ):
            return flip_bits(tensor, n=4, seed=self._rng.randrange(1 << 30))
        return tensor
