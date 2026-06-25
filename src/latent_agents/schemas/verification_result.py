"""Verification result schema (§6.3).

Each Verifier returns a `VerificationResult` whose `test_id` maps back to an
FR-ID in the requirements doc (e.g. "FR-LG-001").
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

_FR_ID_RE = re.compile(r"^(FR|NFR)-[A-Z]{2}-\d{3}$")


class Verdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"


class Evidence(BaseModel):
    """Pointers to snapshots / logs that justify the verdict."""

    model_config = ConfigDict(frozen=True)

    snapshots: list[str] = Field(default_factory=list)
    logs: list[str] = Field(default_factory=list)


class VerificationResult(BaseModel):
    """The outcome of a single verifier run against one target artifact."""

    model_config = ConfigDict(frozen=True)

    test_id: str = Field(description="FR-XX-NNN or NFR-XX-NNN")
    target: str = Field(description="ID of the artifact being verified")
    verdict: Verdict
    measured_value: Any = None
    threshold: Any = None
    evidence: Evidence = Field(default_factory=Evidence)
    executed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    agent_version: str
    message: str | None = None

    @field_validator("test_id")
    @classmethod
    def _validate_test_id(cls, v: str) -> str:
        if not _FR_ID_RE.match(v):
            raise ValueError(f"test_id must match FR-XX-NNN, got {v!r}")
        return v

    def is_pass(self) -> bool:
        return self.verdict == Verdict.PASS
