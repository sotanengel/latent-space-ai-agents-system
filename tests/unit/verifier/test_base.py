"""Tests for the BaseVerifier abstraction and registry."""

from __future__ import annotations

from typing import Any

import pytest

from latent_agents.schemas import Verdict, VerificationResult
from latent_agents.verifier import BaseVerifier, get_verifier, list_verifiers, register


@register
class _AlwaysPass(BaseVerifier[str]):
    test_id = "FR-LG-001"

    def _evaluate(self, target: str) -> tuple[Verdict, str | None]:
        return Verdict.PASS, None


@register
class _BoundedNorm(BaseVerifier[float]):
    """Toy verifier: norm must be in [0.5, 2.0]."""

    test_id = "FR-LG-003"
    threshold: tuple[float, float] = (0.5, 2.0)

    def _evaluate(self, target: float) -> tuple[Verdict, str | None]:
        lo, hi = self.threshold
        if lo <= target <= hi:
            return Verdict.PASS, None
        return Verdict.FAIL, f"norm {target} outside [{lo}, {hi}]"

    def _measured(self, target: float) -> Any:  # type: ignore[override]
        return target


class TestBaseVerifier:
    def test_pass_returns_verification_result(self) -> None:
        v = _AlwaysPass()
        r = v.verify("anything", target_id="latent-1")
        assert isinstance(r, VerificationResult)
        assert r.test_id == "FR-LG-001"
        assert r.verdict == Verdict.PASS
        assert r.target == "latent-1"

    def test_fail_carries_message_and_measured(self) -> None:
        v = _BoundedNorm()
        r = v.verify(5.0, target_id="latent-2")
        assert r.verdict == Verdict.FAIL
        assert r.measured_value == 5.0
        assert r.threshold == (0.5, 2.0)
        assert r.message is not None
        assert "outside" in r.message

    def test_test_id_required(self) -> None:
        class _Bad(BaseVerifier[int]):  # type: ignore[misc]
            # missing test_id
            def _evaluate(self, target: int) -> tuple[Verdict, str | None]:
                return Verdict.PASS, None

        with pytest.raises(TypeError):
            _Bad()


class TestRegistry:
    def test_registered_verifiers_are_discoverable(self) -> None:
        names = list_verifiers()
        assert "FR-LG-001" in names
        assert "FR-LG-003" in names

    def test_get_by_test_id(self) -> None:
        cls = get_verifier("FR-LG-001")
        assert cls is _AlwaysPass

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            get_verifier("FR-XX-999")
