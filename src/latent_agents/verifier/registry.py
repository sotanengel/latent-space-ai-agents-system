"""Verifier registry: discover verifiers by FR-ID."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from .base import BaseVerifier

_REGISTRY: dict[str, type] = {}

C = TypeVar("C", bound=type)


def register(cls: C) -> C:
    """Class decorator that registers a verifier by its `test_id`."""
    test_id = getattr(cls, "test_id", None)
    if not isinstance(test_id, str):
        raise TypeError(f"{cls.__name__} must declare a string `test_id` to be registered")
    _REGISTRY[test_id] = cls
    return cls


def get_verifier(test_id: str) -> type[BaseVerifier[object]]:
    if test_id not in _REGISTRY:
        raise KeyError(f"no verifier registered for {test_id}")
    return _REGISTRY[test_id]


def list_verifiers() -> list[str]:
    return sorted(_REGISTRY)
