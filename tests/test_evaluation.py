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


# ---------------------------------------------------------------- cross-check
def _always(**kw):
    from receptionist.nlu.llm_extractor import LLMExtraction
    return lambda text, ref, name, phone: LLMExtraction(**kw)


def test_crosscheck_absent_leaves_the_baseline_identical():
    from receptionist.evaluation.corpus import CASES, REFERENCE
    from receptionist.evaluation.harness import evaluate

    base = evaluate(CASES, REFERENCE, 0.30, seed=11)
    same = evaluate(CASES, REFERENCE, 0.30, seed=11, crosscheck=lambda *a: None)
    assert base.by_outcome() == same.by_outcome()
    assert same.crosschecks_unavailable == len(CASES)
    assert same.crosschecks_run == 0


def test_an_unavailable_model_is_counted_not_ignored():
    """A run where the model was down must not look like one where it agreed."""
    from receptionist.evaluation.corpus import CASES, REFERENCE
    from receptionist.evaluation.harness import evaluate

    def broken(text, ref, name, phone):
        raise TimeoutError("ollama down")

    r = evaluate(CASES, REFERENCE, 0.0, seed=11, crosscheck=broken)
    assert r.crosschecks_unavailable == len(CASES)
    assert r.rescued == 0


def test_agreement_does_not_change_any_outcome():
    from receptionist.evaluation.corpus import CASES, REFERENCE
    from receptionist.evaluation.harness import evaluate

    base = evaluate(CASES, REFERENCE, 0.0, seed=11)
    agreeing = evaluate(CASES, REFERENCE, 0.0, seed=11,
                        crosscheck=_always(procedure=None, appointment_datetime=None))
    assert base.by_outcome() == agreeing.by_outcome()
    assert agreeing.crosschecks_run == len(CASES)


def test_a_dissenting_model_raises_flagged_count_and_never_silent_count():
    """Whatever the second extractor says, silent errors cannot go up."""
    from receptionist.evaluation.corpus import CASES, REFERENCE
    from receptionist.evaluation.harness import evaluate

    base = evaluate(CASES, REFERENCE, 0.0, seed=11)
    dissent = evaluate(CASES, REFERENCE, 0.0, seed=11,
                       crosscheck=_always(procedure="whitening"))

    base_counts, new_counts = base.by_outcome(), dissent.by_outcome()
    assert new_counts["wrong_silent"] <= base_counts["wrong_silent"]
    flagged_before = base_counts["correct_flagged"] + base_counts["wrong_caught"]
    flagged_after = new_counts["correct_flagged"] + new_counts["wrong_caught"]
    assert flagged_after > flagged_before


def test_false_alarms_and_rescues_are_counted_separately():
    """Disagreeing about a correct value is a cost, not a save."""
    from receptionist.evaluation.corpus import CASES, REFERENCE
    from receptionist.evaluation.harness import evaluate

    r = evaluate(CASES, REFERENCE, 0.0, seed=11,
                 crosscheck=_always(procedure="whitening"))
    # At zero ASR error the rules are right about procedure, so every
    # disagreement here is by definition a false alarm.
    assert r.disagreed_on_correct > 0
    assert r.rescued == 0
    assert r.false_alarm_rate == 1.0
