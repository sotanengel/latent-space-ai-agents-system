"""Analysis layer.

Phase 1: TensorInspector verifiers.
Phase 2: ProcrustesAligner + RealignmentValidator suite.
Phase 7: DriftDetector (stub).
"""

from .drift_detector import DriftDetector
from .realignment_validator import (
    ConditionNumberVerifier,
    OrthogonalityVerifier,
    ProcrustesAligner,
    RealignmentValidator,
    RoundTripVerifier,
    SingularValueVerifier,
)
from .tensor_inspector import (
    BatchIndependenceVerifier,
    DistributionVerifier,
    DtypeVerifier,
    NanInfVerifier,
    NormStabilityVerifier,
    ShapeVerifier,
    SparsityVerifier,
)

__all__ = [
    "BatchIndependenceVerifier",
    "ConditionNumberVerifier",
    "DistributionVerifier",
    "DriftDetector",
    "DtypeVerifier",
    "NanInfVerifier",
    "NormStabilityVerifier",
    "OrthogonalityVerifier",
    "ProcrustesAligner",
    "RealignmentValidator",
    "RoundTripVerifier",
    "ShapeVerifier",
    "SingularValueVerifier",
    "SparsityVerifier",
]
