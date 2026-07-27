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
) -> EvaluationReport:
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

        slots = extract_slots(transcript, reference_time, confidences)

        for slot_name, expected in case.expected.items():
            slot = getattr(slots, slot_name)
            actual = slot.value
            if isinstance(expected, datetime) and isinstance(actual, datetime):
                correct = expected == actual
            else:
                correct = str(actual) == str(expected)
            report.outcomes.append(SlotOutcome(
                slot=slot_name, expected=expected, actual=actual,
                correct=correct, confidence=slot.confidence,
                flagged_for_readback=slot.needs_confirmation or not slot.filled,
            ))

    return report


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
