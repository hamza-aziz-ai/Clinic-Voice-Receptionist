"""Pseudonymisation before a transcript leaves the machine.

WHY SURROGATES AND NOT MASKS

``gpt-oss:120b-cloud`` is a remote model - Ollama forwards the prompt to
ollama.com. A dental clinic's transcript is a patient's name, mobile number
and medical procedure in one sentence, so it does not go off-machine intact.

The naive version replaces identifiers with ``<NAME>`` and ``<PHONE>``. That
destroys the very thing the extractor is being asked to reason about: an
utterance reading "my name is <NAME> I need a cleaning" gives the model
nothing to segment, and its answer becomes a restatement of our own mask.

So identifiers are replaced with **surrogates of the same shape and script** -
a Malayalam name becomes a different Malayalam name, a UAE mobile becomes a
different valid UAE mobile. The model sees a well-formed utterance and does
the same segmentation work it would do on the real one. The mapping back is
local and exact.

WHAT THIS HONESTLY COSTS

The surrogate is chosen by us, so whatever the model returns for those two
fields is a value we injected. Agreement on ``patient_name`` and ``phone`` is
therefore guaranteed and carries no information - it says the model can copy,
not that the digits are right. The cross-check has real teeth only on
``appointment_time`` and ``procedure``, which are not redacted because a
procedure name and a clock time are not identifiers.

That is a genuine loss on the two fields where a second opinion would be worth
most, and it is the price of the privacy decision rather than an oversight.
It is asserted in the tests so nobody later reads the cross-check as stronger
than it is.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

# Surrogates are grouped by script so a Malayalam name is replaced by a
# Malayalam one. Substituting a Latin name into an Indic utterance would
# change the language-detection evidence and the segmentation problem, which
# is the opposite of what a faithful surrogate is for.
SURROGATE_NAMES: dict[str, tuple[str, ...]] = {
    "latin": ("Rahul Sharma", "Meera Iyer", "Arjun Das", "Fatima Noor"),
    "ta": ("கார்த்திக் ராமன்", "மீனா செல்வம்"),
    "kn": ("ಸುರೇಶ್ ಕುಮಾರ್", "ಲತಾ ರಾವ್"),
    "ml": ("അനിൽ കുമാർ", "സീത മേനോൻ"),
    "hi": ("राहुल शर्मा", "मीरा अय्यर"),
}

# Valid-looking UAE and India mobiles that are not real allocations to any
# individual we are dealing with. Shape matters: the model should see a
# number its tokeniser treats like the original.
SURROGATE_PHONES: tuple[str, ...] = (
    "+971501110001", "+971551110002", "+919812340001", "+919912340002",
)

# Ranges reused from language detection. Never use isalpha() or \w here -
# Indic combining vowel signs and the virama are categories Mn/Mc and both
# return False, which is the bug that mangled names once already.
_SCRIPT_RANGES: dict[str, tuple[int, int]] = {
    "ta": (0x0B80, 0x0BFF),
    "kn": (0x0C80, 0x0CFF),
    "ml": (0x0D00, 0x0D7F),
    "hi": (0x0900, 0x097F),
}


@dataclass
class Redaction:
    """A transcript safe to send, plus the map that undoes it."""

    text: str
    # surrogate -> original. Never serialised, never logged, never sent.
    reverse: dict[str, str] = field(default_factory=dict)
    # Identifiers the caller asked to remove that were not found verbatim in
    # the transcript. Non-empty means the redaction is incomplete and the
    # text must not be sent - see the note on `redact`.
    unremoved: list[str] = field(default_factory=list)

    def restore(self, value: str | None) -> str | None:
        """Map a model answer back to the real value.

        Exact match only. A model that returns a surrogate with different
        spacing or casing has not identified our value, and guessing at a
        fuzzy match here would silently substitute a real patient's name for
        something the model half-recognised.
        """
        if value is None:
            return None
        return self.reverse.get(value.strip(), value)


def _pick(pool: tuple[str, ...], original: str) -> str:
    """Deterministic surrogate choice.

    Python's built-in hash() is salted per process, so the same transcript
    would get a different surrogate on every run - untraceable in logs and
    impossible to write a stable test against. blake2b is stable across
    processes and machines.
    """
    digest = hashlib.blake2b(original.encode("utf-8"), digest_size=8).digest()
    return pool[int.from_bytes(digest, "big") % len(pool)]


def script_of(text: str) -> str:
    """Which surrogate pool a span should be drawn from."""
    for code, (low, high) in _SCRIPT_RANGES.items():
        if any(low <= ord(ch) <= high for ch in text):
            return code
    return "latin"


def redact(
    transcript: str,
    name: str | None = None,
    phone: str | None = None,
) -> Redaction:
    """Replace the given identifiers with surrogates of the same shape.

    ``name`` and ``phone`` must be the **literal substrings** as they appear
    in the transcript, which is not always what the rule extractor's Slot
    carries. ``NAME_SPAN`` matches up to forty characters after "my name is"
    and is trimmed down by normalisation afterwards, so its ``raw_text`` is
    "Priya Menon I need a cleaning tomorrow a" while its ``value`` is the
    actual name. Redacting on the span replaced most of the sentence with a
    surrogate and destroyed the utterance the model was meant to read. Pass
    the normalised name and the raw phone span.

    An identifier that is not found verbatim is recorded in ``unremoved``
    rather than ignored. Silently skipping it is the single failure in this
    module that cannot be walked back: a real patient name posted to a third
    party, with no digits present for the residual scan to catch.
    """
    text = transcript
    reverse: dict[str, str] = {}
    unremoved: list[str] = []

    for original, pool in (
        (name, None),
        (phone, SURROGATE_PHONES),
    ):
        if not original:
            continue
        original = original.strip()
        if not original:
            continue
        if original not in text:
            unremoved.append(original)
            continue
        chosen = pool if pool is not None else SURROGATE_NAMES[script_of(original)]
        surrogate = _pick(chosen, original)
        text = text.replace(original, surrogate)
        reverse[surrogate] = original

    return Redaction(text=text, reverse=reverse, unremoved=unremoved)


# Anything that survives redaction and still looks like a contact detail is a
# leak. Checked as a last line of defence rather than trusted, because the
# cost of being wrong is a patient's mobile number on someone else's server.
_RESIDUAL_DIGITS = re.compile(r"(?:\+?\d[\d\s\-]{7,}\d)")
_RESIDUAL_SPOKEN = re.compile(
    r"(?:\b(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|double|triple)\b"
    r"[\s,\-]*){7,}",
    re.IGNORECASE,
)


def residual_identifiers(text: str, allowed: set[str]) -> list[str]:
    """Digit runs in ``text`` that are not one of our own surrogates.

    ``allowed`` is the surrogate set. A hit means the rule extractor did not
    find a number that is nonetheless present - a second number in the
    utterance, a number stated before the name, a format the regex missed -
    and the caller must refuse to send rather than send and hope.
    """
    found: list[str] = []
    for pattern in (_RESIDUAL_DIGITS, _RESIDUAL_SPOKEN):
        for match in pattern.finditer(text):
            span = match.group(0).strip()
            compact = re.sub(r"[\s\-]", "", span)
            if any(compact == re.sub(r"[\s\-]", "", a) for a in allowed):
                continue
            found.append(span)
    return found
