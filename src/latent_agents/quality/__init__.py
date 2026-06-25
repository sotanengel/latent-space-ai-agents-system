"""Quality assurance.

Phase 3: LLMJudgeOrchestrator (FR-TA-001..003).
Phase 5: PerformanceProfiler + chaos primitives.
Phase 6: AdversarialTester (still stub on main; lands in Phase 6 PR).
"""

from .adversarial_tester import AdversarialTester
from .chaos import ChaosInjector, FailureMode, delay_call, flip_bits
from .llm_judge_orchestrator import (
    EnsembleResult,
    Judge,
    JudgeVerdict,
    LLMJudgeOrchestrator,
)
from .performance_profiler import PerformanceProfiler, PhaseBreakdown, ProfileSummary

__all__ = [
    "AdversarialTester",
    "ChaosInjector",
    "EnsembleResult",
    "FailureMode",
    "Judge",
    "JudgeVerdict",
    "LLMJudgeOrchestrator",
    "PerformanceProfiler",
    "PhaseBreakdown",
    "ProfileSummary",
    "delay_call",
    "flip_bits",
]
