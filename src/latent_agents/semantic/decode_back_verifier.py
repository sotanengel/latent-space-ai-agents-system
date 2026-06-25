"""Decode-back verifier - Phase 3 stub.

Requirement: FR-SM-001 (semantic equivalence via LLM-as-Judge >= 0.90),
             FR-SM-003 (reverse decoding BLEU >= 0.70 / ROUGE-L >= 0.75)
Tracking issue: #4
"""

from __future__ import annotations


class DecodeBackVerifier:
    """Verify latent-space inference matches text-based inference after decoding."""

    def __init__(self) -> None:
        raise NotImplementedError("Phase 3: see issue #4")
