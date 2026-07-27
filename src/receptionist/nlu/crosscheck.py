"""Reconcile the rule extractor with the LLM's second opinion.

THE ASYMMETRY THAT MAKES THIS SAFE

Disagreement lowers confidence. Agreement never raises it.

The tempting version is symmetric - two extractors agree, so trust the value
more. It is wrong, and it is wrong in the direction that hurts. Both
extractors read the *same* degraded transcript, so their errors are
correlated: when ASR turns "five" into "nine", both read nine and both agree,
confidently, on a wrong number. A symmetric rule would take that agreement as
evidence and push the slot over its threshold, booking silently. That is
precisely the failure this repository is built to prevent, arrived at by
adding a safety feature.

So agreement buys nothing and disagreement costs. The worst case of this
design is a read-back that was not strictly necessary - a question asked of a
caller who is on the line anyway. The worst case of the symmetric design is a
wrong appointment nobody was asked about.

The mechanism is the ceiling already used for Bolna's confidence labels, for
the same reason: a value that can only move one way cannot be gamed into
booking something.

WHAT IT ACTUALLY CHECKS

Only ``appointment_time`` and ``procedure`` carry information. ``patient_name``
and ``phone`` are replaced with surrogates before the transcript leaves the
machine, so the model returns the value we injected and agreement on them is
guaranteed. Comparing them would manufacture a reassuring number that means
nothing, so they are not compared at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .llm_extractor import LLMExtraction
from .slots import SlotSet

# Ceiling applied to a slot the two extractors disagree about. Sits below
# every threshold in CONFIRMATION_THRESHOLDS, so a disagreement always forces
# the read-back rather than merely nudging the number down.
DISAGREEMENT_CEILING = 0.60

# Fields the cross-check can say anything about. Redacted fields are excluded
# by construction - see the module docstring.
COMPARABLE_SLOTS = ("appointment_time", "procedure")

# Two clock times this close are the same appointment. The LLM resolves
# relative dates itself and can land a minute off on "half past three"
# without disagreeing about anything a patient would notice.
DATETIME_TOLERANCE_S = 60


@dataclass
class CrossCheckReport:
    compared: list[str] = field(default_factory=list)
    disagreed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    available: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "compared": self.compared,
            "disagreed": self.disagreed,
            "skipped": self.skipped,
        }


def _same_datetime(rule_value: Any, llm_value: str | None) -> bool | None:
    """True/False, or None when there is nothing to compare."""
    if rule_value is None or not llm_value:
        return None
    try:
        parsed = datetime.fromisoformat(llm_value.strip().replace("Z", ""))
    except ValueError:
        # The model was asked for ISO 8601 and produced something else. That
        # is a failure to answer, not a disagreement about the appointment -
        # penalising the slot for it would punish the caller for our prompt.
        return None
    if not isinstance(rule_value, datetime):
        return None
    return abs((parsed - rule_value).total_seconds()) <= DATETIME_TOLERANCE_S


def _same_procedure(rule_value: Any, llm_value: str | None) -> bool | None:
    if rule_value is None or not llm_value:
        return None
    return str(rule_value).strip().lower() == llm_value.strip().lower()


def apply_crosscheck(
    slots: SlotSet,
    extraction: LLMExtraction | None,
) -> CrossCheckReport:
    """Lower the confidence of any slot the two extractors disagree about.

    Mutates ``slots`` in place and returns what happened, so the console and
    the evaluation harness can show why a read-back was triggered rather than
    presenting an unexplained drop in confidence.
    """
    if extraction is None:
        # No second opinion available. The rule extractor's confidence stands
        # exactly as it was - the system behaves as though this module were
        # not installed, which is the required behaviour when Ollama is down.
        return CrossCheckReport(available=False)

    report = CrossCheckReport()

    for name in COMPARABLE_SLOTS:
        slot = getattr(slots, name)
        if not slot.filled:
            report.skipped.append(name)
            continue
        if slot.confirmed:
            # The caller said it aloud and agreed. A model's opinion does not
            # outrank the person whose appointment it is.
            report.skipped.append(name)
            continue

        llm_value = extraction.value_for(name)
        if name == "appointment_time":
            verdict = _same_datetime(slot.value, llm_value)
        else:
            verdict = _same_procedure(slot.value, llm_value)

        if verdict is None:
            report.skipped.append(name)
            continue

        report.compared.append(name)
        if verdict:
            # Agreement deliberately does nothing. See the module docstring.
            continue

        report.disagreed.append(name)
        if slot.confidence > DISAGREEMENT_CEILING:
            slot.confidence = DISAGREEMENT_CEILING
        slot.notes.append(
            f"second extractor read {llm_value!r}; confidence capped pending read-back"
        )

    return report
