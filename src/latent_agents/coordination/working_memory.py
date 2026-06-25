"""Shared latent working memory + auditor (FR-WM-001..009).

Multi-version, LRU + TTL eviction, refcount-protected. Audit traverses the
store and reports versions per key and any metadata gaps.

This is in-memory only; persistence backends would slot in behind the
same interface in a follow-up phase.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping


@dataclass(frozen=True)
class MemoryEntry:
    payload: bytes
    actor: str
    ts: float
    ttl_seconds: float | None


@dataclass(frozen=True)
class MemoryHandle:
    """Returned to callers — does not expose internal store state."""

    key: str
    version: int
    actor: str
    ts: float


@dataclass
class AuditReport:
    consistent: bool
    versions: dict[str, int] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


class WorkingMemory:
    """In-memory multi-version store with LRU + TTL eviction and refcount pins."""

    def __init__(self, *, capacity: int | None = None) -> None:
        self._capacity = capacity
        self._store: dict[str, list[MemoryEntry]] = {}
        # LRU order over keys
        self._order: OrderedDict[str, None] = OrderedDict()
        self._refcounts: dict[str, int] = {}

    # ------------------------------------------------------------------ put
    def put(
        self,
        key: str,
        payload: bytes,
        *,
        actor: str,
        ttl_seconds: float | None = None,
    ) -> MemoryHandle:
        if not actor:
            raise ValueError("actor must be non-empty")
        now = time.monotonic()
        entry = MemoryEntry(payload=payload, actor=actor, ts=now, ttl_seconds=ttl_seconds)
        versions = self._store.setdefault(key, [])
        versions.append(entry)
        self._order.pop(key, None)
        self._order[key] = None
        self._evict_if_needed()
        return MemoryHandle(key=key, version=len(versions) - 1, actor=actor, ts=now)

    # ------------------------------------------------------------------ get
    def get(self, key: str, *, version: int | None = None) -> MemoryEntry:
        self._expire_if_due(key)
        if key not in self._store:
            raise KeyError(key)
        self._order.pop(key, None)
        self._order[key] = None  # touch for LRU
        versions = self._store[key]
        idx = -1 if version is None else version
        return versions[idx]

    # --------------------------------------------------------------- delete
    def delete(self, key: str) -> None:
        if self._refcounts.get(key, 0) > 0:
            raise RuntimeError(f"cannot delete pinned key {key!r}")
        self._store.pop(key, None)
        self._order.pop(key, None)

    # ------------------------------------------------------------ refcount
    def acquire(self, key: str) -> None:
        if key not in self._store:
            raise KeyError(key)
        self._refcounts[key] = self._refcounts.get(key, 0) + 1

    def release(self, key: str) -> None:
        n = self._refcounts.get(key, 0)
        if n <= 0:
            raise RuntimeError(f"release without acquire on {key!r}")
        if n == 1:
            self._refcounts.pop(key)
        else:
            self._refcounts[key] = n - 1

    # ---------------------------------------------------------- internals
    def _expire_if_due(self, key: str) -> None:
        versions = self._store.get(key)
        if not versions:
            return
        now = time.monotonic()
        latest = versions[-1]
        if latest.ttl_seconds is not None and now - latest.ts > latest.ttl_seconds:
            self._store.pop(key, None)
            self._order.pop(key, None)

    def _evict_if_needed(self) -> None:
        if self._capacity is None:
            return
        while len(self._order) > self._capacity:
            for candidate in list(self._order):
                if self._refcounts.get(candidate, 0) == 0:
                    self._order.pop(candidate)
                    self._store.pop(candidate, None)
                    break
            else:
                # All entries pinned — cannot evict, give up.
                return

    # introspection ------------------------------------------------------
    @property
    def keys(self) -> Mapping[str, list[MemoryEntry]]:
        return self._store


class WorkingMemoryAuditor:
    """Read-only consistency audit over a `WorkingMemory`."""

    def __init__(self, memory: WorkingMemory) -> None:
        self.memory = memory

    def audit(self) -> AuditReport:
        report = AuditReport(consistent=True)
        for key, versions in self.memory.keys.items():
            report.versions[key] = len(versions)
            for i, e in enumerate(versions):
                if not e.actor:
                    report.consistent = False
                    report.issues.append(f"{key}@{i}: empty actor")
                if e.ttl_seconds is not None and e.ttl_seconds < 0:
                    report.consistent = False
                    report.issues.append(f"{key}@{i}: negative TTL")
        return report
