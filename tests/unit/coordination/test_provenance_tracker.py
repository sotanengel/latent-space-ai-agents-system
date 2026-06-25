"""Tests for ProvenanceTracker (FR-PV-001..007)."""

from __future__ import annotations

import pytest

from latent_agents.coordination import HmacSigner, ProvenanceTracker
from latent_agents.schemas import EventType


class TestProvenanceTracker:
    def test_record_and_replay(self) -> None:
        tr = ProvenanceTracker()
        tr.record(event_type=EventType.GENERATE, actor="a", outputs=["l1"])
        tr.record(event_type=EventType.CONSUME, actor="b", inputs=["l1"])
        replay = tr.replay()
        assert len(replay) == 2
        assert replay[0].actor == "a"

    def test_lineage_traverses_dependencies(self) -> None:
        tr = ProvenanceTracker()
        tr.record(event_type=EventType.GENERATE, actor="a", outputs=["l1"])
        tr.record(event_type=EventType.TRANSFORM, actor="b", inputs=["l1"], outputs=["l2"])
        tr.record(event_type=EventType.TRANSFORM, actor="c", inputs=["l2"], outputs=["l3"])
        lineage = tr.lineage_of("l3")
        # l3 derives from l2 which derives from l1
        assert [r.outputs for r in lineage] == [["l1"], ["l2"], ["l3"]]

    def test_chain_tamper_is_detected(self) -> None:
        tr = ProvenanceTracker()
        tr.record(event_type=EventType.GENERATE, actor="a", outputs=["l1"])
        tr.record(event_type=EventType.CONSUME, actor="b", inputs=["l1"])
        assert tr.verify()
        tr.chain.records[0] = tr.chain.records[0].model_copy(update={"actor": "attacker"})
        assert not tr.verify()


class TestHmacSigner:
    def test_sign_and_verify_round_trip(self) -> None:
        signer = HmacSigner(b"secret-key")
        payload = b"hello world"
        sig = signer.sign(payload)
        assert signer.verify(payload, sig)

    def test_tampered_payload_fails(self) -> None:
        signer = HmacSigner(b"secret-key")
        sig = signer.sign(b"hello")
        assert not signer.verify(b"hello?", sig)

    def test_wrong_key_fails(self) -> None:
        s1 = HmacSigner(b"one")
        s2 = HmacSigner(b"two")
        with pytest.raises(ValueError):
            s2.verify(b"x", "not-a-valid-hex-sig")
        assert not s2.verify(b"x", s1.sign(b"x"))


class TestProvenanceSigning:
    def test_signed_record_carries_signature(self) -> None:
        tr = ProvenanceTracker(signer=HmacSigner(b"k"))
        rec = tr.record(event_type=EventType.GENERATE, actor="a", outputs=["l1"])
        assert rec.signature is not None
        assert tr.verify_signatures()

    def test_signature_tamper_detected(self) -> None:
        tr = ProvenanceTracker(signer=HmacSigner(b"k"))
        tr.record(event_type=EventType.GENERATE, actor="a", outputs=["l1"])
        tr.chain.records[0] = tr.chain.records[0].model_copy(update={"actor": "evil"})
        assert not tr.verify_signatures()
