"""Provenance schema (§6.2).

Append-only hash chain over `ProvenanceRecord`s. Used by Phase 4
ProvenanceTracker, but the schema and chain primitives live here so that
Phase 1 capture / serializer code can already emit records.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class EventType(StrEnum):
    GENERATE = "generate"
    TRANSFER = "transfer"
    TRANSFORM = "transform"
    CONSUME = "consume"
    DELETE = "delete"


class Transformation(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class ProvenanceRecord(BaseModel):
    """Single immutable provenance event."""

    model_config = ConfigDict(frozen=True)

    record_id: UUID = Field(default_factory=uuid4)
    event_type: EventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    actor: str
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    transformation: Transformation | None = None
    signature: str | None = None
    previous_record_hash: str | None = None

    def with_signature(self, signature: str) -> ProvenanceRecord:
        return self.model_copy(update={"signature": signature})

    def with_previous_hash(self, previous: str | None) -> ProvenanceRecord:
        return self.model_copy(update={"previous_record_hash": previous})

    def content_hash(self) -> str:
        """Stable hash over content excluding the signature.

        The signature itself is excluded so signing is well-defined.
        """
        payload = self.model_dump_json(exclude={"signature"})
        return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


class _AppendOnlyList(list[ProvenanceRecord]):
    """Subclass that rejects non-`ProvenanceRecord` appends.

    `list.append` accepts anything via the C API, so we override it; we
    cannot prevent pydantic-internal slot assignments without making the
    chain itself frozen, but this catches the common foot-gun.
    """

    def append(self, value: ProvenanceRecord) -> None:
        if not isinstance(value, ProvenanceRecord):
            raise TypeError(f"only ProvenanceRecord can be appended, got {type(value).__name__}")
        super().append(value)


class ProvenanceChain(BaseModel):
    """Append-only chain of provenance records."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    records: _AppendOnlyList = Field(default_factory=_AppendOnlyList)

    def append(self, record: ProvenanceRecord) -> ProvenanceRecord:
        prev = self.records[-1].content_hash() if self.records else None
        linked = record.with_previous_hash(prev)
        self.records.append(linked)
        return linked

    def hash_of(self, record: ProvenanceRecord) -> str:
        return record.content_hash()

    def verify(self) -> bool:
        prev: str | None = None
        for rec in self.records:
            if rec.previous_record_hash != prev:
                return False
            prev = rec.content_hash()
        return True

    def __iter__(self) -> Iterator[ProvenanceRecord]:  # type: ignore[override]
        return iter(self.records)
