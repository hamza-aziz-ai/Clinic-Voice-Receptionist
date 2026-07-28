"""Slot extraction with per-field confidence.

THE CENTRAL DISCIPLINE OF THIS PROJECT

A voice receptionist fails differently from a chatbot. The failure is not
"the model said something wrong" - it is "the model heard something wrong and
booked on it with complete confidence". A misheard digit sends the
confirmation WhatsApp to a stranger. A misheard date books the wrong day. The
caller hangs up believing they have an appointment.

So confidence is tracked per slot, not per utterance, and it is derived from
things that are actually measurable:

  * ASR word-level confidence for the span the value came from
  * whether the value survived normalisation into a valid form
  * whether it was stated once or repeated consistently
  * whether it is intrinsically risky (phone digits are far easier to
    mishear than a procedure name drawn from a closed set)

Anything below threshold is not guessed and is not discarded - it is read
back to the caller for explicit confirmation. That is the entire safety
mechanism, and it is why this is not a Zapier chain.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from .normalize import (
    normalise_datetime,
    normalise_name,
    normalise_phone,
)

SlotName = Literal["patient_name", "phone", "appointment_time", "procedure"]

# Confidence floors below which a slot must be confirmed aloud before use.
# Phone is highest because a single wrong digit is silently catastrophic:
# the number remains valid, so nothing downstream can detect the error.
CONFIRMATION_THRESHOLDS: dict[str, float] = {
    "phone": 0.92,
    "appointment_time": 0.85,
    "patient_name": 0.75,
    "procedure": 0.70,
}

PROCEDURES: dict[str, tuple[str, ...]] = {
    "cleaning": ("cleaning", "scaling", "polish", "hygiene", "clean"),
    "extraction": ("extraction", "remove", "pull", "take out"),
    "root_canal": ("root canal", "rct", "endo"),
    "filling": ("filling", "cavity", "restoration"),
    "checkup": ("checkup", "check up", "consultation", "look at", "examine", "pain"),
    "whitening": ("whitening", "bleaching", "whiten"),
    "braces": ("braces", "aligner", "orthodontic", "invisalign"),
}

PROCEDURE_DURATION_MIN: dict[str, int] = {
    "cleaning": 30, "extraction": 45, "root_canal": 90,
    "filling": 45, "checkup": 20, "whitening": 60, "braces": 45,
}

# How callers actually answer "what would you like to come in for?". They do
# not name procedures - they describe what hurts. A real transcript:
#
#   "I think my wisdom tooth is not coming up properly and its aching my left
#    side of the jaw down to the neck."
#
# matched nothing, and the agent asked the same question four more times.
#
# Every one of these maps to CHECKUP and never to a treatment. A caller
# describing symptoms has not consented to an extraction, and no amount of
# keyword matching can distinguish a wisdom tooth that needs removing from
# one that needs an X-ray. Booking 45 minutes for an extraction off "aching"
# would be the system inventing a clinical decision from a phone call.
SYMPTOM_TERMS: tuple[str, ...] = (
    "aching", "ache", "aches", "hurts", "hurting", "sore", "throbbing",
    "swollen", "swelling", "bleeding", "sensitive", "sensitivity",
    "wisdom tooth", "wisdom teeth", "abscess", "infected", "infection",
    "broken", "chipped", "cracked", "loose", "stuck", "toothache",
)

# Confidence multiplier for a procedure inferred from a symptom rather than
# named. Chosen to land the slot below the 0.70 threshold, so it is always
# read back as "shall I book you a check-up" rather than silently booked.
# The caller said what hurts; they did not say what appointment they wanted.
SYMPTOM_CONFIDENCE_FACTOR = 0.75


@dataclass
class Slot:
    name: SlotName
    raw_text: str | None = None
    value: Any = None
    confidence: float = 0.0
    source: str = "unfilled"
    confirmed: bool = False
    notes: list[str] = field(default_factory=list)
    # The day, when the caller gave one but no time of day. The slot stays
    # unfilled - a date is not an appointment - but the day is not thrown
    # away either, so the follow-up can ask "what time on Saturday?" instead
    # of starting again. See extract_slots.
    pending_date: Any = None

    @property
    def filled(self) -> bool:
        return self.value is not None

    @property
    def needs_confirmation(self) -> bool:
        """Below threshold and not yet confirmed aloud."""
        if not self.filled:
            return False
        if self.confirmed:
            return False
        return self.confidence < CONFIRMATION_THRESHOLDS[self.name]

    @property
    def usable(self) -> bool:
        return self.filled and not self.needs_confirmation

    def confirm(self) -> None:
        """Caller explicitly agreed to the read-back."""
        self.confirmed = True
        self.notes.append("confirmed by caller read-back")

    def reject(self) -> None:
        """Caller said the read-back was wrong - discard, do not keep a guess."""
        self.value = None
        self.raw_text = None
        self.confidence = 0.0
        self.confirmed = False
        self.source = "rejected"
        self.notes.append("rejected at read-back, cleared")


@dataclass
class SlotSet:
    patient_name: Slot = field(default_factory=lambda: Slot("patient_name"))
    phone: Slot = field(default_factory=lambda: Slot("phone"))
    appointment_time: Slot = field(default_factory=lambda: Slot("appointment_time"))
    procedure: Slot = field(default_factory=lambda: Slot("procedure"))

    def all_slots(self) -> list[Slot]:
        return [self.patient_name, self.phone, self.appointment_time, self.procedure]

    @property
    def missing(self) -> list[Slot]:
        return [s for s in self.all_slots() if not s.filled]

    @property
    def pending_confirmation(self) -> list[Slot]:
        return [s for s in self.all_slots() if s.needs_confirmation]

    @property
    def bookable(self) -> bool:
        """Every slot filled AND every slot either confident or confirmed."""
        return all(s.usable for s in self.all_slots())


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------
PHONE_SPAN = re.compile(
    r"((?:\+?\d[\d\s\-]{7,}\d)|(?:(?:\b(?:zero|oh|one|two|three|four|five|six|"
    r"seven|eight|nine|double|triple)\b[\s,\-]*){7,}))",
    re.IGNORECASE,
)
NAME_SPAN = re.compile(
    r"(?:my name is|this is|i am|i'm|name is)\s+([^\d,.;]{2,40})", re.IGNORECASE
)
TIME_SPAN = re.compile(
    r"((?:today|tomorrow|day after tomorrow|next\s+\w+day|monday|tuesday|wednesday|"
    r"thursday|friday|saturday|sunday|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)"
    r"[^,.;]{0,30})",
    re.IGNORECASE,
)


def _asr_confidence(word_confidences: dict[str, float] | None, span: str) -> float:
    """Mean ASR confidence over the words in a span.

    Real ASR returns per-word confidence. Averaging over the span that
    produced a value is what makes slot-level confidence meaningful rather
    than a single number for the whole utterance.
    """
    if not word_confidences:
        return 0.85          # no ASR metadata: assume decent but not certain
    words = re.findall(r"[\w']+", span.lower())
    scores = [word_confidences[w] for w in words if w in word_confidences]
    return sum(scores) / len(scores) if scores else 0.85


def _replaces(slot: Slot, new_confidence: float) -> bool:
    """May a freshly extracted value take this slot?

    Without this check every slot was overwritten by whatever the latest
    utterance happened to match. A real call captured "Hamza Aziz" correctly,
    then the caller said "I am having ache in my left jaw" - which the name
    pattern also matches - and the good name was silently replaced. Nothing
    was flagged, because the replacement scored the same 0.77 the original
    did.

    A caller states each detail once. A later sentence that happens to fit
    the pattern is far more likely to be a false positive than a correction,
    and genuine corrections do not arrive this way: they come through a
    rejected read-back, which clears the slot first.
    """
    if slot.confirmed:
        return False
    if not slot.filled:
        return True
    # Filled and already trusted - leave it alone.
    if not slot.needs_confirmation:
        return False
    # Filled but below threshold: a clearer restatement may still improve it,
    # which is how a caller repeating a bad number gets a better reading.
    return new_confidence > slot.confidence


def extract_slots(
    transcript: str,
    reference_time: datetime,
    word_confidences: dict[str, float] | None = None,
    existing: SlotSet | None = None,
) -> SlotSet:
    """Fill what this utterance supports. Never overwrite a confirmed slot."""
    slots = existing or SlotSet()

    # -- phone -------------------------------------------------------------
    if not slots.phone.confirmed:
        m = PHONE_SPAN.search(transcript)
        if m:
            span = m.group(1).strip()
            parsed = normalise_phone(span)
            if parsed:
                conf = _asr_confidence(word_confidences, span) - parsed.confidence_penalty
                # Digits carry no redundancy - no language model can catch a
                # wrong one - so ASR confidence is discounted for phone spans.
                conf *= 0.95
                conf = max(0.0, min(1.0, conf))
                if _replaces(slots.phone, conf):
                    slots.phone = Slot(
                        "phone", raw_text=span, value=parsed.e164,
                        confidence=conf, source="asr",
                        notes=[f"country {parsed.country}"]
                        + (["digit pattern not recognised for IN/AE"]
                           if parsed.confidence_penalty else []),
                    )

    # -- name --------------------------------------------------------------
    if not slots.patient_name.confirmed:
        m = NAME_SPAN.search(transcript)
        if m:
            span = m.group(1).strip()
            parsed = normalise_name(span)
            if parsed:
                conf = _asr_confidence(word_confidences, span)
                # Proper nouns are out-of-vocabulary for most ASR and are the
                # single most misrecognised field in clinic calls.
                conf = max(0.0, min(1.0, conf * 0.90))
                if _replaces(slots.patient_name, conf):
                    slots.patient_name = Slot(
                        "patient_name", raw_text=span, value=parsed,
                        confidence=conf, source="asr",
                    )

    # -- appointment time --------------------------------------------------
    if not slots.appointment_time.confirmed:
        pending = slots.appointment_time.pending_date
        m = TIME_SPAN.search(transcript)
        span = m.group(1).strip() if m else None

        # A bare time answering "what time on Saturday?" has no day token in
        # it, so TIME_SPAN cannot see it at all. Reattach the day we already
        # understood and let the same parser handle it.
        parse_input = span
        if span is None and pending is not None:
            parse_input = f"{pending.day:02d}/{pending.month:02d} {transcript.strip()}"

        parsed = normalise_datetime(parse_input, reference_time) if parse_input else None

        if parsed and not _replaces(slots.appointment_time, 0.0 if parsed.time_was_vague
                                    else _asr_confidence(word_confidences, span or transcript)):
            parsed = None       # already have a time we trust; do not restate it

        if parsed and parsed.time_was_vague:
            # "Saturday morning" is a day, not an appointment. The old code
            # filled the slot with 10:00 and read it back as "That's Saturday
            # 01 August at 10:00 AM. Shall I book that?" - proposing an hour
            # the caller never said and inviting a yes to it. A yes there
            # books a real chair at a time nobody chose.
            #
            # The slot stays unfilled so the collect loop asks for the time,
            # and the day is kept so the question can be specific.
            slots.appointment_time = Slot(
                "appointment_time", raw_text=span, value=None,
                confidence=0.0, source="date_only",
                notes=["day understood; time of day not stated"],
                pending_date=parsed.when.date(),
            )
        elif parsed:
            conf = _asr_confidence(word_confidences, span or transcript)
            notes = []
            if parsed.when < reference_time:
                conf *= 0.5
                notes.append("resolved to a past time")
            slots.appointment_time = Slot(
                "appointment_time", raw_text=span or transcript.strip(),
                value=parsed.when,
                confidence=max(0.0, min(1.0, conf)), source="asr", notes=notes,
            )

    # -- procedure ---------------------------------------------------------
    if not slots.procedure.confirmed:
        lowered = transcript.lower()
        named = None
        for code, keywords in PROCEDURES.items():
            hit = next((k for k in keywords if k in lowered), None)
            if hit:
                named = (code, hit)
                break

        if named:
            code, hit = named
            conf = max(0.0, min(1.0, _asr_confidence(word_confidences, hit)))
            # Closed vocabulary: a near-miss still lands on a valid value,
            # so this is the most reliable slot on the call.
            if _replaces(slots.procedure, conf):
                slots.procedure = Slot(
                    "procedure", raw_text=hit, value=code,
                    confidence=conf, source="asr",
                    notes=[f"matched keyword {hit!r}"],
                )
        else:
            symptom = next((s for s in SYMPTOM_TERMS if s in lowered), None)
            if symptom:
                # Inferred, not stated. Deliberately scored below threshold so
                # it is read back before anything is booked.
                conf = _asr_confidence(word_confidences, symptom) * SYMPTOM_CONFIDENCE_FACTOR
                conf = max(0.0, min(1.0, conf))
                if _replaces(slots.procedure, conf):
                    slots.procedure = Slot(
                        "procedure", raw_text=symptom, value="checkup",
                        confidence=conf, source="symptom",
                        notes=[f"inferred from symptom {symptom!r}; "
                               "not stated by caller"],
                    )

    return slots


# A value the agent inferred from its own question rather than one the caller
# announced. Scored below the plain-statement confidence so it is read back:
# "Amna Ansari" answering "what is your name?" is almost certainly a name,
# and "almost certainly" is exactly what the read-back gate exists for.
CONTEXT_CONFIDENCE_FACTOR = 0.88

# Replies that answer the question without being the value. Interpreting
# "yes" as a patient called Yes is worse than asking again.
_NON_ANSWERS = {
    "yes", "yeah", "yep", "no", "nope", "ok", "okay", "sure", "please",
    "correct", "right", "wrong", "thanks", "thank you", "hi", "hello",
}


def extract_for_slot(
    text: str,
    slot_name: str,
    reference_time: datetime,
    word_confidences: dict[str, float] | None = None,
) -> Slot | None:
    """Read a bare reply as the slot the agent just asked for.

    THE BUG THIS FIXES

    ``extract_slots`` only finds a name behind a trigger - "my name is",
    "this is", "I'm". A caller answering "Could I take your full name
    please?" with "Amna Ansari" matched nothing, so the agent asked again,
    and again, and then escalated. The caller had answered correctly twice.

    The agent knows what it just asked for. Using that is not a guess, it is
    the most ordinary fact available about the turn, and ignoring it made the
    system deaf to the plainest possible answer.

    Confidence is discounted and the value is therefore read back. The caller
    said a name-shaped thing in reply to a question about names; that is
    strong evidence, not proof.
    """
    stripped = (text or "").strip()
    if not stripped or stripped.lower().strip(".,!?") in _NON_ANSWERS:
        return None

    conf = _asr_confidence(word_confidences, stripped) * CONTEXT_CONFIDENCE_FACTOR

    if slot_name == "patient_name":
        # A bare date is a far likelier reply to a mis-ordered conversation
        # than a patient named "Friday Evening".
        if normalise_datetime(stripped, reference_time):
            return None
        parsed = normalise_name(stripped)
        if not parsed:
            return None
        return Slot("patient_name", raw_text=stripped, value=parsed,
                    confidence=max(0.0, min(1.0, conf * 0.90)), source="answer",
                    notes=["given in answer to a direct question"])

    if slot_name == "phone":
        parsed = normalise_phone(stripped)
        if not parsed:
            return None
        return Slot("phone", raw_text=stripped, value=parsed.e164,
                    confidence=max(0.0, min(1.0, (conf - parsed.confidence_penalty) * 0.95)),
                    source="answer",
                    notes=[f"country {parsed.country}",
                           "given in answer to a direct question"])

    if slot_name == "appointment_time":
        parsed = normalise_datetime(stripped, reference_time)
        if not parsed or parsed.time_was_vague:
            return None
        return Slot("appointment_time", raw_text=stripped, value=parsed.when,
                    confidence=max(0.0, min(1.0, conf)), source="answer",
                    notes=["given in answer to a direct question"])

    return None


def readback_prompt(slot: Slot, language: str = "en") -> str:
    """The question the agent asks to confirm a low-confidence slot.

    Read-backs are per-slot and specific. "Did I get that right?" after a
    four-field summary produces a yes that means nothing - the caller cannot
    hold four values in mind and check each one.
    """
    if slot.name == "phone":
        return f"Let me confirm your number: {spoken_number(slot.value)}. Is that correct?"
    if slot.name == "appointment_time":
        return (
            f"That's {slot.value:%A %d %B} at {slot.value:%I:%M %p}. "
            f"Shall I book that?"
        )
    if slot.name == "patient_name":
        return f"I have your name as {slot.value}. Did I get that right?"
    if slot.source == "symptom":
        # Never read an inferred procedure back as though the caller chose it.
        # "You'd like a checkup" invites a yes to something they never said,
        # and a yes here is what books the appointment.
        return (
            "It sounds like you need someone to take a look at that. "
            "Shall I book you in for a check-up?"
        )
    return f"You'd like a {str(slot.value).replace('_', ' ')}. Is that right?"


def spoken_number(e164: str) -> str:
    """Group a number so a caller can actually check it.

    Read as one run - "9 1 8 4 4 7 6 4 4 1 8 8" - twelve digits are
    unverifiable by ear, which makes the read-back theatre: the caller says
    yes because they lost track, not because it was right. Grouping is the
    difference between confirming and appearing to confirm.
    """
    digits = str(e164 or "").lstrip("+")
    if not digits:
        return ""

    country, rest = "", digits
    for code in ("971", "91"):
        if digits.startswith(code):
            country, rest = code, digits[len(code):]
            break

    groups: list[str] = []
    while len(rest) > 4:
        groups.append(rest[:3])
        rest = rest[3:]
    if rest:
        groups.append(rest)

    return " ".join(([f"plus {country}"] if country else []) + groups)
