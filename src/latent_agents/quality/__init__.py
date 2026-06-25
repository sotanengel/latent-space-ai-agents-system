"""Quality assurance.

Phase 3: LLMJudgeOrchestrator + Judge ensemble (FR-TA-001..003).
Phase 5: PerformanceProfiler + chaos primitives.
Phase 6: AdversarialTester suite (FR-RB-001..007).
"""

from .adversarial_tester import (
    AdversarialTester,
    FGSMAttack,
    NoiseRobustnessVerifier,
    PoisoningDetector,
    PromptInjectionDetector,
    StatisticalAnomalyDetector,
    Watermarker,
)
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
    "FGSMAttack",
    "FailureMode",
    "Judge",
    "JudgeVerdict",
    "LLMJudgeOrchestrator",
    "NoiseRobustnessVerifier",
    "PerformanceProfiler",
    "PhaseBreakdown",
    "PoisoningDetector",
    "ProfileSummary",
    "PromptInjectionDetector",
    "StatisticalAnomalyDetector",
    "Watermarker",
    "delay_call",
    "flip_bits",
]
