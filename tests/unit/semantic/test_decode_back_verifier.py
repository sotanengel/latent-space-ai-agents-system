"""Tests for DecodeBackVerifier text-similarity metrics (FR-SM-003)."""

from __future__ import annotations

from latent_agents.schemas import Verdict
from latent_agents.semantic import DecodeBackVerifier, bleu_score, rouge_l


class TestBleuRouge:
    def test_identical_texts_score_one(self) -> None:
        assert bleu_score("the cat sat", "the cat sat") == 1.0
        assert rouge_l("the cat sat", "the cat sat") == 1.0

    def test_disjoint_texts_score_zero(self) -> None:
        assert bleu_score("aaa bbb", "ccc ddd") == 0.0
        assert rouge_l("aaa bbb", "ccc ddd") == 0.0

    def test_partial_overlap_is_intermediate(self) -> None:
        r = rouge_l("the cat sat on the mat", "the cat lay on the mat")
        assert 0.5 < r < 1.0


class TestDecodeBackVerifier:
    def test_pass_when_similarity_above_threshold(self) -> None:
        v = DecodeBackVerifier(min_bleu=0.5, min_rouge=0.5)
        r = v.verify(("the cat sat on the mat", "the cat sat on the mat"))
        assert r.verdict == Verdict.PASS

    def test_fail_when_unrelated(self) -> None:
        v = DecodeBackVerifier(min_bleu=0.3, min_rouge=0.3)
        r = v.verify(("hello world", "completely different text"))
        assert r.verdict == Verdict.FAIL
