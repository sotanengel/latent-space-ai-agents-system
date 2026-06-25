"""Tests for ProvenanceRecord schema (§6.2)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from latent_agents.schemas import EventType, ProvenanceChain, ProvenanceRecord


class TestProvenanceRecord:
    def test_minimal(self) -> None:
        rec = ProvenanceRecord(
            event_type=EventType.GENERATE,
            actor="agent-a",
            outputs=["latent-1"],
        )
        assert rec.event_type == EventType.GENERATE
        assert rec.previous_record_hash is None  # first record in chain

    def test_signature_optional_until_seal(self) -> None:
        rec = ProvenanceRecord(
            event_type=EventType.CONSUME,
            actor="agent-b",
            inputs=["latent-1"],
        )
        assert rec.signature is None
        rec_sealed = rec.with_signature("sig-bytes")
        assert rec_sealed.signature == "sig-bytes"

    def test_event_type_must_be_valid(self) -> None:
        with pytest.raises(ValidationError):
            ProvenanceRecord(
                event_type="explode",  # type: ignore[arg-type]
                actor="x",
            )


class TestProvenanceChain:
    def test_append_links_previous_hash(self) -> None:
        chain = ProvenanceChain()
        r1 = chain.append(
            ProvenanceRecord(event_type=EventType.GENERATE, actor="a", outputs=["l1"])
        )
        r2 = chain.append(ProvenanceRecord(event_type=EventType.CONSUME, actor="b", inputs=["l1"]))
        assert r1.previous_record_hash is None
        assert r2.previous_record_hash == chain.hash_of(r1)

    def test_chain_verify_detects_tamper(self) -> None:
        chain = ProvenanceChain()
        chain.append(ProvenanceRecord(event_type=EventType.GENERATE, actor="a", outputs=["l1"]))
        chain.append(ProvenanceRecord(event_type=EventType.CONSUME, actor="b", inputs=["l1"]))
        assert chain.verify()
        # tamper with first record's actor field
        chain.records[0] = chain.records[0].model_copy(update={"actor": "attacker"})
        assert not chain.verify()

    def test_append_only(self) -> None:
        chain = ProvenanceChain()
        chain.append(ProvenanceRecord(event_type=EventType.GENERATE, actor="a", outputs=["l1"]))
        with pytest.raises(TypeError):
            chain.records.append("not a record")  # type: ignore[arg-type]
