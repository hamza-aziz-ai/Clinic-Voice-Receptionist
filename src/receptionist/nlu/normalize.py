"""Normalisation of spoken values into bookable data.

Callers do not speak in database formats. They say "double nine" for 99,
"tomorrow morning", "next Tuesday", and read phone numbers in groups. Every
one of those has to become an unambiguous value or be rejected outright -
"probably the 15th" is not something you book a clinic appointment on.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

SPOKEN_DIGITS = {
    "zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
MULTIPLIERS = {"double": 2, "triple": 3}

INDIA_MOBILE = re.compile(r"^(?:\+?91)?([6-9]\d{9})$")
UAE_MOBILE = re.compile(r"^(?:\+?971|0)?(5[0245689]\d{7})$")


@dataclass(frozen=True)
class NormalisedPhone:
    e164: str
    country: str
    confidence_penalty: float = 0.0


def normalise_phone(raw: str) -> NormalisedPhone | None:
    """Spoken digit sequence to E.164, or None if it cannot be trusted.

    Returning None is the whole point. A phone number that is nearly right is
    worse than no phone number: the confirmation WhatsApp goes to a stranger
    and the patient never learns their appointment exists.
    """
    if not raw:
        return None

    tokens = re.split(r"[\s,\-]+", raw.lower().strip())
    digits: list[str] = []
    pending_multiplier = 1

    for token in tokens:
        token = token.strip(".")
        if token in MULTIPLIERS:
            pending_multiplier = MULTIPLIERS[token]
            continue
        if token in SPOKEN_DIGITS:
            digits.append(SPOKEN_DIGITS[token] * pending_multiplier)
            pending_multiplier = 1
            continue
        cleaned = re.sub(r"[^\d+]", "", token)
        if cleaned:
            digits.append(cleaned * pending_multiplier if pending_multiplier > 1 else cleaned)
            pending_multiplier = 1

    joined = "".join(digits)
    if not joined:
        return None

    plus = joined.startswith("+")
    bare = joined.lstrip("+")

    m = UAE_MOBILE.match(bare)
    if m:
        return NormalisedPhone(f"+971{m.group(1)}", "AE")
    m = INDIA_MOBILE.match(bare)
    if m:
        return NormalisedPhone(f"+91{m.group(1)}", "IN")

    # Right length but wrong shape - a plausible mishearing, not a valid number.
    if 9 <= len(bare) <= 13:
        return NormalisedPhone(f"+{bare}" if plus else bare, "UNKNOWN",
                               confidence_penalty=0.4)
    return None


WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
TIME_WORDS = {
    "morning": time(10, 0), "afternoon": time(15, 0),
    "evening": time(18, 0), "noon": time(12, 0),
}


@dataclass(frozen=True)
class NormalisedDateTime:
    when: datetime
    was_relative: bool
    time_was_vague: bool


def normalise_datetime(raw: str, reference: datetime) -> NormalisedDateTime | None:
    """Spoken date/time to a concrete datetime, relative to the call time."""
    if not raw:
        return None
    text = raw.lower().strip()

    target_date: date | None = None
    relative = False

    if "today" in text:
        target_date, relative = reference.date(), True
    elif "day after tomorrow" in text:
        target_date, relative = reference.date() + timedelta(days=2), True
    elif "tomorrow" in text:
        target_date, relative = reference.date() + timedelta(days=1), True
    else:
        for name, idx in WEEKDAYS.items():
            if name in text:
                ahead = (idx - reference.weekday()) % 7
                if ahead == 0 or "next" in text:
                    ahead += 7 if ahead == 0 else 0
                target_date, relative = reference.date() + timedelta(days=ahead or 7), True
                break

    if target_date is None:
        m = re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", text)
        if m:
            day, month = int(m.group(1)), int(m.group(2))
            year = int(m.group(3) or reference.year)
            if year < 100:
                year += 2000
            try:
                target_date = date(year, month, day)
            except ValueError:
                return None
            # Consume the date before parsing time. Otherwise the time regex
            # matches the day-of-month first: "15/08 at 11am" parsed as 15:00
            # rather than 11:00, silently booking four hours late.
            text = text[: m.start()] + " " + text[m.end() :]

    if target_date is None:
        return None

    vague = False
    hhmm = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", text)
    when_time: time | None = None
    for word, t in TIME_WORDS.items():
        if word in text:
            when_time, vague = t, True
            break
    if when_time is None and hhmm:
        hour = int(hhmm.group(1))
        minute = int(hhmm.group(2) or 0)
        meridiem = hhmm.group(3)
        if meridiem == "pm" and hour < 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        elif meridiem is None and hour < 8:
            hour += 12          # "at 3" in a clinic means 15:00
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            when_time = time(hour, minute)
    if when_time is None:
        when_time, vague = time(10, 0), True

    return NormalisedDateTime(
        datetime.combine(target_date, when_time), relative, vague
    )


def spoken_time(t: time) -> str:
    """A time of day the way a receptionist says it out loud.

    Every hour spoken to a caller needs its half of the day attached. The
    agent said "We're open from 9 in the morning until 8", which is 8 pm and
    reads as 8 am - and an agent that is unclear about opening hours will be
    told to come in at a time the clinic is shut.

    24-hour format is unambiguous but wrong for speech: a TTS voice reading
    "20:00" is not how anyone says it, and "closing at 20:00" is harder to
    act on than "closing at 8 in the evening".
    """
    hour12 = t.hour % 12 or 12
    base = f"{hour12}:{t.minute:02d}" if t.minute else f"{hour12}"

    if t.hour == 12 and t.minute == 0:
        return "12 noon"
    if t.hour < 12:
        part = "in the morning"
    elif t.hour < 17:
        part = "in the afternoon"
    else:
        part = "in the evening"
    return f"{base} {part}"


NAME_NOISE = re.compile(
    r"\b(my name is|this is|i am|i'm|it's|speaking|name|call me)\b", re.IGNORECASE
)

# Words that end a name. Without these the capture runs on into the rest of
# the sentence - "Priya Menon I Need A Cleaning" scored 0.77 and passed the
# 0.75 confidence gate, so it was a SILENT error rather than a caught one.
NAME_BOUNDARY = {
    "and", "i", "we", "my", "me", "can", "could", "need", "needs", "want",
    "wants", "would", "like", "please", "for", "the", "a", "an", "is", "am",
    "have", "has", "book", "booking", "appointment", "calling", "call",
    "tomorrow", "today", "next", "on", "at", "with", "about", "regarding",
}

# The same bug again, one vocabulary further out. "my name is Sara Ali root
# canal on 15/08" stopped at "on" - which is in the list - having already
# swallowed "root canal", which was not. The name became "Sara Ali Root
# Canal" at confidence 0.765, above the 0.75 gate, so it booked and went out
# on the WhatsApp confirmation without ever being read back: a silent error,
# the one class this system claims none of.
#
# It survived because every corpus utterance happens to put a stop-word
# between the name and the procedure ("I need a cleaning"). A caller who
# does not pause that way was never tested. Found when the LLM cross-check's
# redaction guard refused to send, because the bloated name was not a
# verbatim substring of the transcript.
#
# Kept in sync with PROCEDURES by test_name_boundary_covers_every_procedure_word,
# which imports both - a runtime import would be circular, since slots.py
# imports this module.
CLINICAL_BOUNDARY = {
    "cleaning", "scaling", "polish", "hygiene", "clean",
    "extraction", "remove", "pull", "take", "out",
    "root", "canal", "rct", "endo",
    "filling", "cavity", "restoration",
    "checkup", "check", "up", "consultation", "look", "examine", "pain",
    "whitening", "bleaching", "whiten",
    "braces", "aligner", "orthodontic", "invisalign",
    # Symptom words too - callers say "my name is Sara Ali wisdom tooth
    # aching", and a name that swallows the symptom is the same bug as a name
    # that swallows the procedure. Kept in sync by the same test.
    "aching", "ache", "aches", "hurts", "hurting", "sore", "throbbing",
    "swollen", "swelling", "bleeding", "sensitive", "sensitivity",
    "wisdom", "tooth", "teeth", "abscess", "infected", "infection",
    "broken", "chipped", "cracked", "loose", "stuck", "toothache",
}
NAME_BOUNDARY |= CLINICAL_BOUNDARY

MAX_NAME_WORDS = 4


def normalise_name(raw: str) -> str | None:
    if not raw:
        return None
    cleaned = NAME_NOISE.sub(" ", raw)
    # Explicitly permit the Indic blocks. Python's \w does NOT match
    # combining vowel signs or the virama (categories Mn/Mc), so a plain
    # \w-based filter silently strips them: "അഞ്ജലി" became "അഞ ജല".
    # Same root cause as the language-detection bug - Unicode marks are not
    # letters, and Indic scripts are largely built from them.
    cleaned = re.sub(r"[^\w\s'\-.\u0900-\u0D7F]", " ", cleaned, flags=re.UNICODE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .-'")

    kept: list[str] = []
    for word in cleaned.split():
        if word.lower() in NAME_BOUNDARY:
            break
        kept.append(word)
        if len(kept) >= MAX_NAME_WORDS:
            break
    cleaned = " ".join(kept)

    if not cleaned or len(cleaned) < 2:
        return None
    # Title-case only ASCII; Indic scripts have no case and must be left alone.
    if cleaned.isascii():
        cleaned = " ".join(w.capitalize() for w in cleaned.split())
    return cleaned
