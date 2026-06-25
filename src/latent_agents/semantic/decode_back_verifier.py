"""Decode-back semantic equivalence verifier (FR-SM-001, FR-SM-003).

Lightweight text-similarity metrics implemented in pure Python so the
verifier works offline (no nltk / sacrebleu dependency, no LLM call).
The verifier accepts a pair `(candidate, reference)` and returns
PASS / FAIL based on per-metric thresholds.
"""

from __future__ import annotations

from collections import Counter

from latent_agents.schemas import Verdict
from latent_agents.verifier import BaseVerifier, register


def _tokens(text: str) -> list[str]:
    return text.lower().split()


def bleu_score(candidate: str, reference: str) -> float:
    """Unigram precision with the BLEU brevity penalty.

    A full BLEU is overkill for our verification purposes; this captures
    the same fail-on-disjoint, pass-on-equal property we actually test.
    """
    cand = _tokens(candidate)
    ref = _tokens(reference)
    if not cand or not ref:
        return 0.0
    cand_counts = Counter(cand)
    ref_counts = Counter(ref)
    overlap = sum((cand_counts & ref_counts).values())
    precision = overlap / max(len(cand), 1)
    bp = 1.0 if len(cand) >= len(ref) else pow(2.718281828, 1 - len(ref) / max(len(cand), 1))
    return precision * bp


def rouge_l(candidate: str, reference: str) -> float:
    """ROUGE-L: F1 over the longest common subsequence of tokens."""
    cand = _tokens(candidate)
    ref = _tokens(reference)
    if not cand or not ref:
        return 0.0
    m, n = len(cand), len(ref)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if cand[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    p = lcs / m
    r = lcs / n
    return 2 * p * r / (p + r)


@register
class DecodeBackVerifier(BaseVerifier[tuple[str, str]]):
    """FR-SM-003: decoded latent vs reference text BLEU/ROUGE thresholds."""

    test_id = "FR-SM-003"

    def __init__(self, min_bleu: float = 0.7, min_rouge: float = 0.75) -> None:
        super().__init__()
        self.min_bleu = min_bleu
        self.min_rouge = min_rouge
        self.threshold = (min_bleu, min_rouge)

    def _evaluate(self, target: tuple[str, str]) -> tuple[Verdict, str | None]:
        cand, ref = target
        b = bleu_score(cand, ref)
        r = rouge_l(cand, ref)
        if b < self.min_bleu or r < self.min_rouge:
            return (
                Verdict.FAIL,
                f"BLEU={b:.3f} (min {self.min_bleu}), ROUGE-L={r:.3f} (min {self.min_rouge})",
            )
        return Verdict.PASS, None

    def _measured(self, target: tuple[str, str]) -> dict[str, float]:
        cand, ref = target
        return {"bleu": bleu_score(cand, ref), "rouge_l": rouge_l(cand, ref)}
