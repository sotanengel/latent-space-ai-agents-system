"""Tests for VerificationResult schema (§6.3 of requirements doc)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from latent_agents.schemas import Evidence, Verdict, VerificationResult


class TestVerdict:
    def test_known_verdict_values(self) -> None:
        assert Verdict.PASS.value == "pass"
        assert Verdict.FAIL.value == "fail"
        assert Verdict.WARNING.value == "warning"


class TestVerificationResult:
    def test_minimal_construction(self) -> None:
        result = VerificationResult(
            test_id="FR-LG-001",
            target="latent-abc",
            verdict=Verdict.PASS,
            agent_version="0.1.0",
        )
        assert result.test_id == "FR-LG-001"
        assert result.is_pass()
        assert result.executed_at is not None

    def test_test_id_must_match_fr_pattern(self) -> None:
        with pytest.raises(ValidationError):
            VerificationResult(
                test_id="not-an-fr-id",
                target="x",
                verdict=Verdict.PASS,
                agent_version="0.1.0",
            )

    def test_is_pass_false_on_fail(self) -> None:
        result = VerificationResult(
            test_id="FR-LG-001",
            target="x",
            verdict=Verdict.FAIL,
            measured_value=42,
            threshold=10,
            agent_version="0.1.0",
        )
        assert not result.is_pass()

    def test_evidence_default_empty(self) -> None:
        result = VerificationResult(
            test_id="FR-KV-001",
            target="x",
            verdict=Verdict.PASS,
            agent_version="0.1.0",
        )
        assert result.evidence.snapshots == []
        assert result.evidence.logs == []

    def test_evidence_attach(self) -> None:
        ev = Evidence(snapshots=["snap-1"], logs=["captured 12 layers"])
        result = VerificationResult(
            test_id="FR-KV-002",
            target="kv-xyz",
            verdict=Verdict.PASS,
            evidence=ev,
            agent_version="0.1.0",
        )
        assert result.evidence.snapshots == ["snap-1"]

    def test_serialize_roundtrip(self) -> None:
        result = VerificationResult(
            test_id="FR-LG-004",
            target="x",
            verdict=Verdict.WARNING,
            measured_value=0.5,
            threshold=1.0,
            agent_version="0.1.0",
        )
        data = result.model_dump_json()
        restored = VerificationResult.model_validate_json(data)
        assert restored == result
