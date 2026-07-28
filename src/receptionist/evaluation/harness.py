"""Multilingual slot-extraction evaluation.

THE PROBLEM THIS SOLVES

I do not speak Tamil, Kannada or Malayalam. On a voice project that is not a
footnote - it means I cannot listen to the agent and tell whether it heard a
name correctly, and "it sounded fine" is unavailable to me as a quality signal.

The answer is to stop relying on listening. Every test case pairs an utterance
with the structured values it is supposed to produce. Correctness becomes
"did the extractor return the ground-truth slots", which is checkable without
understanding a word - and is a stronger check than a fluent speaker skimming
transcripts, because it is exhaustive and it runs on every commit.

ASR error injection matters as much as the clean case. Real telephony ASR on
Indic languages is materially worse than on English: 8 kHz audio, code-
switching, and proper nouns that are out of vocabulary. Testing only clean
transcripts measures a system nobody will ever operate.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..nlu.language import detect_language
from ..nlu.slots import extract_slots


@dataclass(frozen=True)
class TestCase:
    case_id: str
    language: str
    transcript: str
    expected: dict[str, Any]
    note: str = ""


@dataclass
class SlotOutcome:
    slot: str
    expected: Any
    actual: Any
    correct: bool
    confidence: float
    flagged_for_readback: bool

    @property
    def outcome(self) -> str:
        """Four-way outcome. The distinction that matters is the last two.

        A wrong value that was flagged for read-back is a caught error - the
        caller gets asked. A wrong value that passed confidently is a silent
        error, and it is the only one that reaches a patient.
        """
        if self.correct and not self.flagged_for_readback:
            return "correct_confident"
        if self.correct:
            return "correct_flagged"
        if self.flagged_for_readback:
            return "wrong_caught"
        return "wrong_silent"


@dataclass
class EvaluationReport:
    outcomes: list[SlotOutcome] = field(default_factory=list)
    language_correct: int = 0
    language_total: int = 0
    # Cross-check bookkeeping. Separated into the two things that matter:
    # a disagreement about a value that was wrong is the mechanism working,
    # a disagreement about a value that was right is the cost of running it.
    crosschecks_run: int = 0
    crosschecks_unavailable: int = 0
    # How many utterances the alternative extractor actually handled, and how
    # many fell back to the rules. Counted rather than assumed: a run where
    # the model was unreachable for half the corpus would otherwise be
    # reported as the model's score, when it is mostly the fallback's.
    extractor_used: int = 0
    extractor_unavailable: int = 0
    disagreed_on_wrong: int = 0
    disagreed_on_correct: int = 0
    rescued: int = 0          # was heading for a silent error, now flagged

    @property
    def false_alarm_rate(self) -> float:
        """Share of disagreements that were about a value the rules got right.

        The honest cost figure. Each one is a question asked of a caller who
        did not need to be asked.
        """
        total = self.disagreed_on_correct + self.disagreed_on_wrong
        return self.disagreed_on_correct / total if total else 0.0

    def by_outcome(self) -> dict[str, int]:
        counts = {k: 0 for k in
                  ("correct_confident", "correct_flagged", "wrong_caught", "wrong_silent")}
        for o in self.outcomes:
            counts[o.outcome] += 1
        return counts

    @property
    def slot_accuracy(self) -> float:
        return sum(o.correct for o in self.outcomes) / len(self.outcomes) if self.outcomes else 0.0

    @property
    def silent_error_rate(self) -> float:
        """The number that actually matters. Wrong AND unflagged."""
        if not self.outcomes:
            return 0.0
        return self.by_outcome()["wrong_silent"] / len(self.outcomes)

    @property
    def language_accuracy(self) -> float:
        return self.language_correct / self.language_total if self.language_total else 0.0


# --------------------------------------------------------------------------
# ASR degradation
# --------------------------------------------------------------------------
DIGIT_CONFUSIONS = {
    "nine": "five", "five": "nine", "four": "one", "one": "nine",
    "eight": "three", "three": "eight", "six": "seven", "seven": "six",
    "two": "zero", "zero": "two",
}


def degrade_transcript(
    text: str, severity: float, rng: random.Random
) -> tuple[str, dict[str, float]]:
    """Simulate telephony ASR error and emit per-word confidence.

    Two failure modes, both real: digits get confused with acoustically close
    digits, and low-confidence words get marked down. Crucially the returned
    confidence correlates with the corruption - which is what lets the
    read-back mechanism catch errors rather than merely detect them after
    the fact.
    """
    words = text.split()
    out: list[str] = []
    confidences: dict[str, float] = {}

    for word in words:
        bare = re.sub(r"[^\w']", "", word.lower())
        corrupt = rng.random() < severity

        if corrupt and bare in DIGIT_CONFUSIONS:
            replacement = DIGIT_CONFUSIONS[bare]
            out.append(word.lower().replace(bare, replacement))
            confidences[replacement] = rng.uniform(0.45, 0.70)
            continue

        if corrupt and len(bare) > 3 and bare.isascii():
            # Proper nouns and long words: keep the token, drop confidence.
            out.append(word)
            confidences[bare] = rng.uniform(0.40, 0.65)
            continue

        out.append(word)
        if bare:
            confidences[bare] = rng.uniform(0.88, 0.99)

    return " ".join(out), confidences


# --------------------------------------------------------------------------
def evaluate(
    cases: list[TestCase],
    reference_time: datetime,
    asr_severity: float = 0.0,
    seed: int = 0,
    crosscheck: Any = None,
    extractor: Any = None,
) -> EvaluationReport:
    """Score the extractor, optionally with the LLM second opinion applied.

    ``crosscheck`` is the same callable the workflow takes:
    ``(text, reference_time, name, phone) -> LLMExtraction | None``. Passing
    one measures the *combined* system, which is the only comparison worth
    making - a second extractor scored on its own says nothing about whether
    the thing that books appointments got safer.

    ``extractor`` replaces the rule extractor entirely:
    ``(text, reference_time, word_confidences) -> SlotSet | None``. Returning
    None falls back to the rules and is counted, because that is exactly what
    the live system does when the model is unreachable - and a run where it
    was down for half the corpus must not be reported as the model's score.

    Until this existed the harness only ever scored the rules, so the accuracy
    table described the fallback rather than the path that actually runs.
    """
    rng = random.Random(seed)
    report = EvaluationReport()

    for case in cases:
        transcript, confidences = (
            degrade_transcript(case.transcript, asr_severity, rng)
            if asr_severity > 0
            else (case.transcript, None)
        )

        detected = detect_language(transcript)
        report.language_total += 1
        if detected.language == case.language:
            report.language_correct += 1

        slots = None
        if extractor is not None:
            try:
                slots = extractor(transcript, reference_time, confidences)
            except Exception:
                slots = None
            if slots is None:
                report.extractor_unavailable += 1
            else:
                report.extractor_used += 1
        if slots is None:
            slots = extract_slots(transcript, reference_time, confidences)

        # Snapshot before the cross-check so its effect is attributable. A
        # slot already flagged by low ASR confidence was not "rescued" by the
        # second extractor, and counting it as such would flatter the feature.
        flagged_before = {
            s.name: (s.needs_confirmation or not s.filled) for s in slots.all_slots()
        }

        if crosscheck is not None:
            _run_crosscheck(report, crosscheck, slots, transcript, reference_time)

        for slot_name, expected in case.expected.items():
            slot = getattr(slots, slot_name)
            actual = slot.value
            if isinstance(expected, datetime) and isinstance(actual, datetime):
                correct = expected == actual
            else:
                correct = str(actual) == str(expected)
            flagged = slot.needs_confirmation or not slot.filled
            report.outcomes.append(SlotOutcome(
                slot=slot_name, expected=expected, actual=actual,
                correct=correct, confidence=slot.confidence,
                flagged_for_readback=flagged,
            ))

            if crosscheck is not None and flagged and not flagged_before[slot_name]:
                # Newly flagged because of the cross-check.
                if correct:
                    report.disagreed_on_correct += 1
                else:
                    # Wrong, and would have gone through unflagged. This is
                    # the only number that justifies running a second model.
                    report.disagreed_on_wrong += 1
                    report.rescued += 1

    return report


def _run_crosscheck(
    report: EvaluationReport,
    crosscheck: Any,
    slots: Any,
    transcript: str,
    reference_time: datetime,
) -> None:
    """Apply the second opinion to ``slots`` in place, recording availability.

    An unavailable cross-check is counted rather than ignored. A run where
    the model was down for half the corpus and silently contributed nothing
    would otherwise be indistinguishable from a run where it agreed with
    everything - and those two say opposite things about the feature.
    """
    from ..nlu.crosscheck import apply_crosscheck

    try:
        extraction = crosscheck(
            transcript, reference_time,
            slots.patient_name.value, slots.phone.raw_text,
        )
    except Exception:
        extraction = None

    result = apply_crosscheck(slots, extraction)
    if result.available:
        report.crosschecks_run += 1
    else:
        report.crosschecks_unavailable += 1


def render_report(report: EvaluationReport, label: str) -> str:
    c = report.by_outcome()
    total = len(report.outcomes)
    return (
        f"{label:<26} slots {report.slot_accuracy:6.1%}   "
        f"lang {report.language_accuracy:6.1%}   "
        f"confident-ok {c['correct_confident']:>3}  "
        f"flagged-ok {c['correct_flagged']:>3}  "
        f"caught {c['wrong_caught']:>3}  "
        f"SILENT {c['wrong_silent']:>3}  "
        f"({report.silent_error_rate:.1%} of {total})"
    )
