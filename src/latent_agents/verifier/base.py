"""Abstract Verifier base class.

Every concrete verifier maps to one requirement ID (FR-XX-NNN) and returns
a `VerificationResult`. Subclasses implement `_evaluate(target)` and
optionally `_measured(target)`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar

from latent_agents import __version__
from latent_agents.schemas import Evidence, Verdict, VerificationResult

T = TypeVar("T")


class BaseVerifier(ABC, Generic[T]):
    """Base class for a single-requirement verifier."""

    test_id: ClassVar[str]
    # Threshold is instance-level so subclasses can configure per-instance bounds.
    threshold: Any = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Concrete subclasses must declare a test_id. Abstract intermediates
        # (still carrying __abstractmethods__) are exempt. Inheritance counts.
        if (
            not cls.__dict__.get("__abstractmethods__")
            and "test_id" not in cls.__dict__
            and not any("test_id" in base.__dict__ for base in cls.__mro__[1:])
        ):
            cls._missing_test_id = True  # type: ignore[attr-defined]

    def __init__(self) -> None:
        if getattr(type(self), "_missing_test_id", False):
            raise TypeError(f"{type(self).__name__} must declare `test_id` (FR-XX-NNN)")

    @abstractmethod
    def _evaluate(self, target: T) -> tuple[Verdict, str | None]:
        """Return verdict and optional human-readable message."""

    def _measured(self, target: T) -> Any:
        """Override to attach a measured value to the result."""
        return None

    def _evidence(self, target: T) -> Evidence:
        return Evidence()

    def verify(self, target: T, *, target_id: str = "<unspecified>") -> VerificationResult:
        verdict, message = self._evaluate(target)
        return VerificationResult(
            test_id=self.test_id,
            target=target_id,
            verdict=verdict,
            measured_value=self._measured(target),
            threshold=self.threshold,
            evidence=self._evidence(target),
            agent_version=__version__,
            message=message,
        )
