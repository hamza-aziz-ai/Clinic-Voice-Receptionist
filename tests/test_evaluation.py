from __future__ import annotations
import pytest
from receptionist.evaluation.corpus import CASES, REFERENCE
from receptionist.evaluation.harness import degrade_transcript, evaluate
import random


class TestCorpus:
    def test_covers_every_supported_language(self):
        assert {c.language for c in CASES} >= {"en", "ta", "kn", "ml", "hi"}

    def test_every_case_declares_ground_truth(self):
        for c in CASES:
            assert c.expected, f"{c.case_id} has no expected values"


class TestEvaluation:
    def test_language_detection_is_perfect_on_clean_transcripts(self):
        assert evaluate(CASES, REFERENCE, 0.0).language_accuracy == 1.0

    def test_no_silent_errors_on_clean_transcripts(self):
        """A wrong value that was not flagged is the only kind that reaches a patient."""
        assert evaluate(CASES, REFERENCE, 0.0).silent_error_rate == 0.0

    @pytest.mark.parametrize("severity", [0.15, 0.30, 0.50])
    def test_no_silent_errors_under_asr_degradation(self, severity):
        """The safety property that matters: errors get caught, not shipped."""
        assert evaluate(CASES, REFERENCE, severity, seed=11).silent_error_rate == 0.0

    def test_degradation_reduces_accuracy_but_increases_catching(self):
        clean = evaluate(CASES, REFERENCE, 0.0, seed=11)
        noisy = evaluate(CASES, REFERENCE, 0.5, seed=11)
        assert noisy.slot_accuracy < clean.slot_accuracy
        assert noisy.by_outcome()["wrong_caught"] > clean.by_outcome()["wrong_caught"]

    def test_clean_accuracy_meets_the_documented_bar(self):
        """Pins the README claim. If this drops, the claim is wrong, not the test."""
        assert evaluate(CASES, REFERENCE, 0.0).slot_accuracy >= 0.90


class TestDegradation:
    def test_is_reproducible(self):
        a = degrade_transcript("nine eight seven six", 0.5, random.Random(1))
        b = degrade_transcript("nine eight seven six", 0.5, random.Random(1))
        assert a == b

    def test_corruption_lowers_reported_confidence(self):
        """Confidence must correlate with corruption or read-back cannot work."""
        _, conf = degrade_transcript(
            "nine eight seven six five four three two one zero", 1.0, random.Random(3))
        assert conf and min(conf.values()) < 0.75

    def test_zero_severity_leaves_text_untouched(self):
        text = "my name is Priya Menon"
        out, _ = degrade_transcript(text, 0.0, random.Random(1))
        assert out == text
