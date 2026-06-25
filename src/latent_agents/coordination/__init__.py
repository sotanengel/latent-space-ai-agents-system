"""Phase 4 coordination layer.

Implemented: WorkingMemory + WorkingMemoryAuditor (FR-WM-001..007),
ProvenanceTracker + HmacSigner (FR-PV-001..007), LatentFlowTracer
(FR-PV-005).
"""

from .latent_flow_tracer import LatentFlowTracer, SpanEvent
from .provenance_tracker import HmacSigner, ProvenanceTracker, Signer
from .working_memory import (
    AuditReport,
    MemoryEntry,
    MemoryHandle,
    WorkingMemory,
    WorkingMemoryAuditor,
)

__all__ = [
    "AuditReport",
    "HmacSigner",
    "LatentFlowTracer",
    "MemoryEntry",
    "MemoryHandle",
    "ProvenanceTracker",
    "Signer",
    "SpanEvent",
    "WorkingMemory",
    "WorkingMemoryAuditor",
]
