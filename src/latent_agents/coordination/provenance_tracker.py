"""ProvenanceTracker + HmacSigner (FR-PV-001..007).

`ProvenanceTracker` wraps the existing `ProvenanceChain` (which already
provides append-only + previous-hash chaining) with: lineage traversal,
replay, and optional HMAC-SHA256 signatures on each record. HMAC is the
right choice here for shared-secret deployments; an Ed25519 signer can
slot in by implementing the same `Signer` protocol.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Any, Protocol

from latent_agents.schemas import EventType, ProvenanceChain, ProvenanceRecord


class Signer(Protocol):
    def sign(self, payload: bytes) -> str: ...
    def verify(self, payload: bytes, signature: str) -> bool: ...


class HmacSigner:
    """HMAC-SHA256 signer using a shared symmetric key."""

    def __init__(self, key: bytes) -> None:
        if not key:
            raise ValueError("HMAC key must be non-empty")
        self._key = key

    def sign(self, payload: bytes) -> str:
        return hmac.new(self._key, payload, hashlib.sha256).hexdigest()

    def verify(self, payload: bytes, signature: str) -> bool:
        # Reject obvious garbage so callers can distinguish a malformed
        # signature from a key/payload mismatch.
        try:
            bytes.fromhex(signature)
        except ValueError as e:
            raise ValueError(f"signature is not valid hex: {e}") from e
        expected = self.sign(payload)
        return hmac.compare_digest(expected, signature)


class ProvenanceTracker:
    """Append-only provenance log with lineage / replay / optional signing."""

    def __init__(self, *, signer: Signer | None = None) -> None:
        self.chain = ProvenanceChain()
        self._signer = signer

    # ------------------------------------------------------------- record
    def record(
        self,
        *,
        event_type: EventType,
        actor: str,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        transformation: Any = None,
    ) -> ProvenanceRecord:
        rec = ProvenanceRecord(
            event_type=event_type,
            actor=actor,
            inputs=inputs or [],
            outputs=outputs or [],
            transformation=transformation,
        )
        linked = self.chain.append(rec)
        if self._signer is not None:
            payload = linked.model_dump_json(exclude={"signature"}).encode()
            signed = linked.with_signature(self._signer.sign(payload))
            self.chain.records[-1] = signed
            return signed
        return linked

    # ------------------------------------------------------------- query
    def replay(self) -> list[ProvenanceRecord]:
        return list(self.chain.records)

    def verify(self) -> bool:
        return self.chain.verify()

    def verify_signatures(self) -> bool:
        if self._signer is None:
            return True
        for rec in self.chain.records:
            if rec.signature is None:
                return False
            payload = rec.model_dump_json(exclude={"signature"}).encode()
            if not self._signer.verify(payload, rec.signature):
                return False
        return True

    def lineage_of(self, latent_id: str) -> list[ProvenanceRecord]:
        """Records that contributed to `latent_id`, in causal order (BFS reversed)."""
        by_output: dict[str, ProvenanceRecord] = {}
        for r in self.chain.records:
            for out in r.outputs:
                by_output[out] = r
        order: list[ProvenanceRecord] = []
        seen: set[str] = set()
        frontier = [latent_id]
        while frontier:
            nxt: list[str] = []
            for lid in frontier:
                rec = by_output.get(lid)
                if rec is None or str(rec.record_id) in seen:
                    continue
                seen.add(str(rec.record_id))
                order.append(rec)
                nxt.extend(rec.inputs)
            frontier = nxt
        return list(reversed(order))
