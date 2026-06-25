"""Tests for WorkingMemory + WorkingMemoryAuditor (FR-WM-001..007)."""

from __future__ import annotations

import time

import pytest

from latent_agents.coordination import MemoryEntry, WorkingMemory, WorkingMemoryAuditor


class TestWorkingMemory:
    def test_put_get_roundtrip_preserves_metadata(self) -> None:
        wm = WorkingMemory()
        h = wm.put("latent-1", b"payload", actor="agent-a")
        assert h.actor == "agent-a"
        loaded = wm.get("latent-1")
        assert loaded.payload == b"payload"
        assert loaded.actor == "agent-a"

    def test_versioning_returns_specific_version(self) -> None:
        wm = WorkingMemory()
        wm.put("k", b"v1", actor="a")
        wm.put("k", b"v2", actor="a")
        assert wm.get("k").payload == b"v2"
        assert wm.get("k", version=0).payload == b"v1"

    def test_missing_key_raises(self) -> None:
        wm = WorkingMemory()
        with pytest.raises(KeyError):
            wm.get("nope")

    def test_dangling_reference_after_delete(self) -> None:
        wm = WorkingMemory()
        wm.put("k", b"v", actor="a")
        wm.delete("k")
        with pytest.raises(KeyError):
            wm.get("k")

    def test_refcount_blocks_eviction(self) -> None:
        wm = WorkingMemory(capacity=2)
        wm.put("a", b"a", actor="x")
        wm.acquire("a")
        wm.put("b", b"b", actor="x")
        wm.put("c", b"c", actor="x")
        # "a" should NOT be evicted because refcount > 0
        assert wm.get("a").payload == b"a"
        wm.release("a")

    def test_lru_evicts_least_recently_used(self) -> None:
        wm = WorkingMemory(capacity=2)
        wm.put("a", b"a", actor="x")
        wm.put("b", b"b", actor="x")
        wm.get("a")  # touch a
        wm.put("c", b"c", actor="x")  # should evict b
        with pytest.raises(KeyError):
            wm.get("b")
        assert wm.get("a").payload == b"a"
        assert wm.get("c").payload == b"c"

    def test_ttl_expiry(self) -> None:
        wm = WorkingMemory()
        wm.put("k", b"v", actor="x", ttl_seconds=0.05)
        time.sleep(0.1)
        with pytest.raises(KeyError):
            wm.get("k")


class TestWorkingMemoryAuditor:
    def test_audit_passes_for_consistent_memory(self) -> None:
        wm = WorkingMemory()
        wm.put("k", b"v", actor="a")
        wm.put("k", b"v2", actor="a")
        rep = WorkingMemoryAuditor(wm).audit()
        assert rep.consistent
        assert rep.versions["k"] == 2

    def test_audit_detects_missing_metadata(self) -> None:
        wm = WorkingMemory()
        h = wm.put("k", b"v", actor="a")
        # Corrupt by replacing the internal entry's actor with empty (only this
        # auditor-level test pokes at the private store).
        wm._store["k"][0] = MemoryEntry(payload=b"v", actor="", ts=h.ts, ttl_seconds=None)
        rep = WorkingMemoryAuditor(wm).audit()
        assert not rep.consistent
