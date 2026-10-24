"""Inbound call state machine.

    greeting → detect_language → collect → confirm ⇄ collect → book → notify → end
                                              ↓
                                          escalate

Escalation is a first-class outcome, not an error path. A receptionist that
cannot understand a caller must hand off to a human, and doing so cleanly is
better product behaviour than booking something plausible. The conditions are
explicit: too many failed read-backs, undetectable language, or repeated
no-availability.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal

from ..config import settings as _settings
from ..messaging.base import MessagingConnector, OutboundMessage, render_template
from ..messaging.dispatch import dispatch_due
from ..nlu.crosscheck import CrossCheckReport, apply_crosscheck
from ..nlu.language import LANGUAGE_NAMES, detect_language
from ..nlu.normalize import spoken_time
from ..nlu.slots import (
    PROCEDURE_DURATION_MIN,
    Slot,
    SlotSet,
    extract_for_slot,
    extract_slots,
    readback_prompt,
)
from ..scheduling.calendar import Calendar

CallState = Literal[
    "greeting", "detect_language", "collect", "confirm",
    "book", "notify", "ended", "escalated",
]

# Conversation limits, tunable per deployment via .env - a clinic with
# patient callers may want more patience than one with trade callers.
MAX_READBACK_FAILURES = _settings.max_readback_failures
MAX_COLLECT_TURNS = 20        # runaway guard only; progress is what matters

# How many times one slot may be asked for before handing off. Repeating an
# identical question is the failure mode this exists to stop: a real call went
#
#   agent : What would you like to come in for?
#   caller: Are you asking about the procedure?
#   agent : What would you like to come in for?
#   caller: I don't understand what you are trying to ask.
#   agent : What would you like to come in for?
#
# A caller who did not understand a sentence will not understand the same
# sentence. Each attempt is worded differently and the third hands to a human.
MAX_ASKS_PER_SLOT = _settings.max_asks_per_slot

# Escalate on turns that add nothing, not on turns elapsed. A caller steadily
# supplying details should never run out of budget; one going in circles
# should. The absolute cap below is only a runaway guard.
MAX_TURNS_WITHOUT_PROGRESS = _settings.max_turns_without_progress

# The caller telling us the agent is not making sense. Worth detecting
# explicitly: the reply is otherwise indistinguishable from silence, and the
# agent answers it by repeating itself, which is what made the loop above feel
# like talking to a wall.
CONFUSION_MARKERS = (
    "don't understand", "dont understand", "do not understand",
    "what do you mean", "are you asking", "not sure what you",
    "come again", "say that again", "repeat that", "didn't catch",
    "didnt catch", "makes no sense", "confused",
)


def sounds_confused(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in CONFUSION_MARKERS)


# Whole words, never substrings. Substring matching read "yesterday" as a yes
# and "I know" as a no - the second rejected and cleared a correct name.
AFFIRMATIVE_WORDS = frozenset(
    {"yes", "yeah", "yep", "correct", "right", "ok", "okay", "sari", "seri", "haan"}
)
# Negation wins over any affirmative word: "that's not right" contains
# "right". Apostrophe-free contractions are listed explicitly; matching any
# word ending in "nt" would make "appointment" and "want" mean no.
NEGATIVE_WORDS = frozenset({
    "no", "not", "wrong", "incorrect", "illai", "illa", "nahi",
    "isnt", "wasnt", "dont", "doesnt", "didnt", "arent", "cant",
})


def _answer_words(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.lower().replace("’", "'"))


def is_negative(text: str) -> bool:
    return any(w in NEGATIVE_WORDS or w.endswith("n't") for w in _answer_words(text))


def is_affirmative(text: str) -> bool:
    return not is_negative(text) and any(w in AFFIRMATIVE_WORDS for w in _answer_words(text))


def _same_words(a: str, b: str) -> bool:
    return " ".join(a.lower().split()) == " ".join(b.lower().split())


def slot_read_back_by(slots: SlotSet, agent_said: str, language: str) -> Slot | None:
    """The one pending slot this agent turn read back, or None.

    A replayed yes or no is only evidence about the question that was actually
    asked. The agent's turn has to be our own read-back for that slot, word for
    word: parsing values out of free agent speech accepted "change your number
    from 0501234567?" as a confirmation of that number, and misread our own
    "That's Tuesday 28 July at 03:00 PM" as a vague morning.
    """
    if not agent_said:
        return None
    matches = [s for s in slots.pending_confirmation
               if _same_words(readback_prompt(s, language), agent_said)]
    return matches[0] if len(matches) == 1 else None


def _carries_new_detail(text: str, now: datetime) -> bool:
    """Does this reply contain booking information rather than a yes or no?

    Run against an empty slot set so it answers "is there anything here at
    all", independent of what the call has already captured.
    """
    if sounds_confused(text):
        return False
    probe = extract_slots(text, now)
    return any(s.filled for s in probe.all_slots())


@dataclass
class Turn:
    speaker: Literal["caller", "agent"]
    text: str
    state: CallState
    note: str = ""


@dataclass
class CallSession:
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    caller_number: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    state: CallState = "greeting"
    language: str = "uncertain"
    language_confidence: float = 0.0
    slots: SlotSet = field(default_factory=SlotSet)
    transcript: list[Turn] = field(default_factory=list)
    booking_id: str | None = None
    escalation_reason: str | None = None
    readback_failures: int = 0
    collect_turns: int = 0
    # How many times each slot has been asked for. Drives the rewording and
    # the hand-off, so the agent cannot ask the same question indefinitely.
    asks: dict[str, int] = field(default_factory=dict)
    messages: list[OutboundMessage] = field(default_factory=list)
    # Upper bound on any slot's confidence for this call. 1.0 means the
    # transcript is the only evidence. A replayed Bolna execution sets this
    # from the extraction confidence labels, because poor audio is a property
    # of the call rather than of one field. It can only ever lower a slot.
    confidence_ceiling: float = 1.0
    # Supplied when the call originates from a source that can redeliver, so
    # a retry reserves the same appointment instead of a second one.
    idempotency_key: str | None = None
    # The slot the agent last asked for, so a bare reply can be read as an
    # answer to that question rather than matched against nothing.
    awaiting: str | None = None
    # Consecutive turns that produced no new detail.
    turns_without_progress: int = 0
    # What the second extractor said last turn, for the console. None means
    # no cross-check ran, which is the default and not a fault.
    crosscheck: CrossCheckReport | None = None
    # A second opinion computed once for the whole call, used instead of
    # calling the model on every turn. Set by the post-call ingest, where the
    # entire transcript is available up front. Measured at 4.8s per model
    # call, a five-turn replay spends 24 seconds in a webhook handler to ask
    # the same question five times about text that is not changing.
    crosscheck_cache: Any = None
    crosscheck_cached: bool = False
    # The slot currently being read back. Cleared whenever the call leaves the
    # confirm state, so a stale reference cannot absorb a later answer.
    pending_slot: Slot | None = None
    # Anything the recogniser wanted flagged about the audio itself - 8 kHz
    # telephony against models benchmarked at 16 kHz, above all. Surfaced
    # rather than swallowed, because it is the gap between a published error
    # rate and the one a clinic actually gets.
    transcript_notes: list[str] = field(default_factory=list)

    def say(self, text: str, note: str = "") -> str:
        self.transcript.append(Turn("agent", text, self.state, note))
        return text

    def hear(self, text: str, note: str = "") -> None:
        self.transcript.append(Turn("caller", text, self.state, note))

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "caller_number": self.caller_number,
            "started_at": self.started_at.isoformat(),
            "state": self.state,
            "language": self.language,
            "language_name": LANGUAGE_NAMES.get(self.language, self.language),
            "language_confidence": round(self.language_confidence, 2),
            "booking_id": self.booking_id,
            "escalation_reason": self.escalation_reason,
            "crosscheck": self.crosscheck.to_dict() if self.crosscheck else None,
            "slots": [
                {
                    "name": s.name,
                    "value": s.value.isoformat() if hasattr(s.value, "isoformat") else s.value,
                    "confidence": round(s.confidence, 2),
                    "confirmed": s.confirmed,
                    "needs_confirmation": s.needs_confirmation,
                    "notes": s.notes,
                }
                for s in self.slots.all_slots()
            ],
            "transcript": [
                {"speaker": t.speaker, "text": t.text, "state": t.state, "note": t.note}
                for t in self.transcript
            ],
            "messages": [
                {"template": m.template, "to": m.to, "language": m.language,
                 "status": m.status, "send_after": m.send_after.isoformat() if m.send_after else None}
                for m in self.messages
            ],
        }


GREETINGS = {
    "en": "Thank you for calling {clinic}. How can I help you today?",
    "ta": "{clinic}-க்கு அழைத்ததற்கு நன்றி. நான் எப்படி உதவ முடியும்?",
    "kn": "{clinic} ಗೆ ಕರೆ ಮಾಡಿದ್ದಕ್ಕೆ ಧನ್ಯವಾದಗಳು. ನಾನು ಹೇಗೆ ಸಹಾಯ ಮಾಡಲಿ?",
    "ml": "{clinic}-ലേക്ക് വിളിച്ചതിന് നന്ദി. എനിക്ക് എങ്ങനെ സഹായിക്കാം?",
    "hi": "{clinic} को कॉल करने के लिए धन्यवाद। मैं आपकी कैसे मदद कर सकता हूँ?",
}


class CallHandler:
    def __init__(
        self,
        calendar: Calendar,
        messaging: MessagingConnector,
        clinic_name: str = "Al Noor Dental",
        review_link: str = "https://g.page/r/alnoor-dental/review",
        crosscheck: Any = None,
        understand: Any = None,
    ) -> None:
        self.calendar = calendar
        self.messaging = messaging
        self.clinic_name = clinic_name
        self.review_link = review_link
        # Callable(transcript, reference_time, name, phone) -> LLMExtraction
        # or None. Injected rather than imported so the workflow layer has no
        # opinion about LangChain, and so the default remains a system with no
        # second extractor at all.
        self.crosscheck = crosscheck
        # Callable(text, now, word_confidences, slots, awaiting) -> SlotSet or
        # None. The primary understanding layer when present; returning None
        # hands the turn to the rule extractor, so an unreachable model costs
        # nuance rather than the ability to answer the phone.
        self.understand = understand

    # ------------------------------------------------------------------
    def start(self, caller_number: str = "") -> CallSession:
        session = CallSession(caller_number=caller_number)
        session.state = "detect_language"
        session.say(GREETINGS["en"].format(clinic=self.clinic_name), "default greeting")
        return session

    def handle_utterance(
        self,
        session: CallSession,
        text: str,
        now: datetime,
        word_confidences: dict[str, float] | None = None,
        *,
        agent_said: str | None = None,
    ) -> str:
        """One caller turn.

        ``agent_said`` is what the agent said just before this turn, supplied
        when replaying a call someone else's agent conducted. ``None`` means
        the live path, where the question was our own read-back.
        """
        session.hear(text)

        if session.state == "detect_language":
            det = detect_language(text)
            session.language = det.language
            session.language_confidence = det.confidence
            if not det.is_confident:
                session.language = "en"
                session.say(
                    "I'll continue in English. " +
                    GREETINGS["en"].format(clinic=self.clinic_name),
                    f"language undetermined ({det.evidence})",
                )
            session.state = "collect"

        if session.state == "confirm":
            return self._handle_readback_answer(
                session, text, now, word_confidences, agent_said)

        if session.state == "collect":
            return self._collect(session, text, now, word_confidences)

        return session.say("Sorry, could you repeat that?")

    # ------------------------------------------------------------------
    def _collect(
        self, session: CallSession, text: str, now: datetime,
        word_confidences: dict[str, float] | None,
    ) -> str:
        session.collect_turns += 1
        before = sum(1 for s in session.slots.all_slots() if s.filled)

        # The model understands the sentence; the rules are the fallback for
        # when it is unreachable. That is the opposite of how this started,
        # and it is the right way round: every extraction bug this system has
        # had was a missing regex, a missing keyword or a missing trigger.
        #
        # Confidence does not come from the model either way - see
        # nlu/llm_slots. It says what was said; the audio says how sure we are.
        understood = None
        if self.understand is not None:
            understood = self.understand(
                text, now, word_confidences, session.slots, session.awaiting
            )
        if understood is not None:
            session.slots = understood
        else:
            session.slots = extract_slots(text, now, word_confidences, session.slots)

        # The agent knows what it just asked for. A bare "Amna Ansari" has no
        # "my name is" in front of it and matched nothing, so the agent asked
        # three times and escalated on a caller who had answered correctly
        # twice. Only consulted when the general extractor found nothing for
        # that slot, so an explicit statement always wins.
        awaiting = session.awaiting
        if awaiting and not getattr(session.slots, awaiting).filled:
            answered = extract_for_slot(text, awaiting, now, word_confidences)
            if answered is not None:
                setattr(session.slots, awaiting, answered)

        self._apply_ceiling(session)
        self._apply_crosscheck(session, text, now)

        # Progress, not elapsed turns, is what decides whether this call is
        # going anywhere. Counting every turn cut off a caller who was
        # steadily supplying details but had lost turns to a question the
        # agent kept failing to understand.
        if sum(1 for s in session.slots.all_slots() if s.filled) > before:
            session.turns_without_progress = 0
        else:
            session.turns_without_progress += 1

        if session.turns_without_progress > MAX_TURNS_WITHOUT_PROGRESS:
            return self._escalate(session, "several turns with no new details")
        if session.collect_turns > MAX_COLLECT_TURNS:
            return self._escalate(session, "call ran too long without completing")

        # Before anything is read back or asked for. A day the clinic will not
        # accept is settled the moment it is understood - whether the caller
        # gave an hour with it or not.
        closed = self._closed_day_reply(session, session.slots.appointment_time, now)
        if closed:
            return closed

        pending = session.slots.pending_confirmation
        if pending:
            session.state = "confirm"
            session.pending_slot = pending[0]
            return session.say(
                readback_prompt(pending[0], session.language),
                f"{pending[0].name} confidence {pending[0].confidence:.2f} below threshold",
            )

        missing = session.slots.missing
        if missing:
            name = missing[0].name
            session.asks[name] = session.asks.get(name, 0) + 1
            attempt = session.asks[name]

            if attempt > MAX_ASKS_PER_SLOT:
                # Asking a fourth time is not going to work, and the caller
                # has already told us twice that the question is unclear.
                return self._escalate(
                    session,
                    f"could not capture {name} after {MAX_ASKS_PER_SLOT} attempts",
                )

            prompt = self._ask_for(name, attempt)
            if name == "appointment_time" and missing[0].pending_date:
                prompt = self._ask_time_on(missing[0].pending_date, attempt)
            if attempt > 1 and sounds_confused(text):
                prompt = f"Sorry, let me put that differently. {prompt}"
            session.awaiting = name
            return session.say(prompt, f"missing {name} (attempt {attempt})")

        return self._book(session, now)

    def _handle_readback_answer(
        self, session: CallSession, text: str, now: datetime,
        word_confidences: dict[str, float] | None,
        agent_said: str | None = None,
    ) -> str:
        slot = session.pending_slot
        session.pending_slot = None
        affirmative = is_affirmative(text)
        negative = is_negative(text)

        if slot is not None and agent_said is not None:
            # Replay: bind the answer to the question the real agent asked,
            # which need not be the slot this state machine picked. A yes or
            # no to anything else is not evidence about any slot - no
            # confirmation, no rejection, no strike. Volunteered details are
            # still collected below.
            slot = slot_read_back_by(session.slots, agent_said, session.language)

        if slot is None:
            session.state = "collect"
            return self._collect(session, text, now, word_confidences)

        if affirmative:
            slot.confirm()
            session.state = "collect"
            session.collect_turns -= 1     # a successful read-back is not a failed turn
            # Confirming a value moves it from unusable to bookable, which is
            # progress even though no new slot was filled. Without this the
            # caller is penalised a turn for cooperating.
            session.turns_without_progress = -1   # _collect increments it back to 0
            return self._collect(session, "", now, word_confidences)

        if negative:
            slot.reject()
            session.readback_failures += 1
            if session.readback_failures > MAX_READBACK_FAILURES:
                return self._escalate(session, "repeated failure to confirm details")
            session.state = "collect"
            # A rejected read-back counts as an ask, so the re-ask is worded
            # differently. Repeating the question the caller just corrected is
            # how a call turns into a loop.
            session.asks[slot.name] = session.asks.get(slot.name, 0) + 1
            return session.say(
                self._ask_for(slot.name, session.asks[slot.name]),
                "read-back rejected, re-asking",
            )

        # The caller answered a different question. Asked to confirm a name,
        # they gave their phone number - and got "Sorry, was that a yes or a
        # no?", then the same again, until the call escalated. People do not
        # stay on the agent's script, and volunteering the next detail is
        # cooperative behaviour, not a failed confirmation.
        #
        # The pending slot stays unconfirmed and will be read back again, so
        # nothing bypasses the gate; what changes is that the turn counts as
        # progress instead of a strike.
        if _carries_new_detail(text, now):
            session.state = "collect"
            return self._collect(session, text, now, word_confidences)

        # Genuinely ambiguous - do not treat silence as yes.
        session.readback_failures += 1
        if session.readback_failures > MAX_READBACK_FAILURES:
            return self._escalate(session, "could not obtain a clear confirmation")
        session.pending_slot = slot
        return session.say("Sorry, was that a yes or a no?", "ambiguous confirmation")

    def _apply_crosscheck(self, session: CallSession, text: str, now: datetime) -> None:
        """Ask the second extractor, and let it lower confidence only.

        Runs after the ceiling and before anything reads
        ``pending_confirmation``, so a disagreement is visible to the same
        booking decision the rule extractor's own confidence feeds.

        Any failure inside leaves the call exactly as it was. The second
        opinion is an enhancement; a caller must not be unable to book a
        cleaning because a language model timed out.
        """
        if self.crosscheck is None and not session.crosscheck_cached:
            return
        try:
            if session.crosscheck_cached:
                extraction = session.crosscheck_cache
            else:
                extraction = self.crosscheck(
                    text, now,
                    session.slots.patient_name.value,
                    session.slots.phone.raw_text,
                )
            report = apply_crosscheck(session.slots, extraction)
        except Exception:
            return
        session.crosscheck = report

    @staticmethod
    def _apply_ceiling(session: CallSession) -> None:
        """Clamp freshly extracted confidences to the call-level ceiling.

        Applied here, immediately after extraction and before anything reads
        ``pending_confirmation``, so the clamp is what the booking decision
        sees. Clamping afterwards would let a slot clear the gate on a
        confidence the call never justified and get capped a turn too late.

        Confirmed slots are left alone: the caller already said the value
        aloud and agreed with it, which outranks a statistical label.
        """
        if session.confidence_ceiling >= 1.0:
            return
        for slot in session.slots.all_slots():
            if not slot.filled or slot.confirmed:
                continue
            if slot.confidence > session.confidence_ceiling:
                slot.confidence = session.confidence_ceiling
                slot.notes.append(
                    f"capped at {session.confidence_ceiling:.2f} by call audio quality"
                )

    # Each slot gets three different wordings, not one repeated three times.
    # The second offers examples, because a caller who did not understand an
    # open question usually can answer a closed one - "are you booking a
    # check-up, a cleaning, or is something hurting?" is answerable even by
    # someone who has no idea what we meant by "come in for".
    ASK_PROMPTS: dict[str, tuple[str, str, str]] = {
        "patient_name": (
            "Could I take your full name please?",
            "Could you give me your first and last name?",
            "I just need a name for the booking - first name is fine.",
        ),
        "phone": (
            "What is the best mobile number to reach you on?",
            "Could you give me a mobile number, starting with the country code?",
            "I still need a contact number - could you read it out digit by digit?",
        ),
        "appointment_time": (
            "What day and time would suit you?",
            "Which day works for you, and roughly what time?",
            "Could you give me a day and a time - for example, Tuesday at 3 pm?",
        ),
        "procedure": (
            "What would you like to come in for?",
            "Are you booking a check-up or a cleaning, or is something hurting?",
            "Is this a routine visit, or is there a problem you want looked at?",
        ),
    }

    def _ask_for(self, slot_name: str, attempt: int = 1) -> str:
        prompts = self.ASK_PROMPTS[slot_name]
        return prompts[min(attempt, len(prompts)) - 1]

    def _closed_day_reply(self, session: CallSession, slot: Any, now: datetime) -> str | None:
        """Say the clinic is shut before asking what time suits on that day.

        A real call went: caller says "Friday evening", agent replies "we're
        open from 9 until 8, what time on Friday 31 July works?", caller picks
        6 pm, agent then refuses because the clinic is closed on Fridays. The
        agent had the opening hours in hand the whole time and still asked the
        caller to choose an hour on a day it would not accept.

        Which is why this reads the *value* as well as `pending_date`. It only
        looked at the latter, so it fired for "Friday morning" and stayed
        silent for "Friday at 10 AM" - the caller who gave a complete answer
        was the one who got no warning, went on to supply a procedure, and was
        refused at booking. The day being unbookable has nothing to do with
        whether an hour came with it.
        """
        day = slot.pending_date or (
            slot.value.date() if slot.filled and slot.value else None)
        if day is None:
            return None
        # Any duration will do - a closed day is closed for all of them.
        probe = datetime.combine(day, self.calendar.hours.open_time)
        if probe.weekday() not in self.calendar.hours.closed_weekdays:
            return None

        # Clear it outright. Leaving a filled value here would send the same
        # closed day round the loop again on the next utterance.
        slot.pending_date = None
        slot.value = None
        slot.confidence = 0.0
        slot.confirmed = False
        session.asks["appointment_time"] = 0
        session.awaiting = "appointment_time"
        alternatives = self.calendar.suggest(
            probe + timedelta(days=1), PROCEDURE_DURATION_MIN.get(
                session.slots.procedure.value or "checkup", 30), count=2)
        offer = ", ".join(f"{a:%A %d %B at %I:%M %p}" for a in alternatives)
        return session.say(
            f"I'm sorry, the clinic is closed on {day:%A}s. "
            + (f"I could offer {offer}. Would either work?" if offer
               else "What other day would suit you?"),
            "requested day is a clinic closure",
        )

    def _ask_time_on(self, day: Any, attempt: int = 1) -> str:
        """Ask for the hour on a day we already have.

        Naming the day matters. "What day and time would suit you?" asked
        after the caller has just said "Saturday morning" reads as though we
        were not listening, and invites them to repeat the day rather than
        supply the missing half.

        The opening hours are read off ClinicHours rather than written into
        the sentence. Hardcoded, they were a second copy of a fact the
        scheduler already owns: change the closing time and the agent keeps
        quoting the old one while the calendar refuses the booking.
        """
        label = f"{day:%A %d %B}"
        hours = self.calendar.hours
        opens, closes = spoken_time(hours.open_time), spoken_time(hours.close_time)
        return (
            f"What time on {label} would suit you?",
            f"We're open from {opens} until {closes}. What time on {label} works?",
            f"Could you give me a time on {label} - for example, 10:30 in the morning?",
        )[min(attempt, 3) - 1]

    # ------------------------------------------------------------------
    def _book(self, session: CallSession, now: datetime) -> str:
        session.state = "book"
        s = session.slots
        result = self.calendar.book(
            patient_name=s.patient_name.value,
            phone=s.phone.value,
            procedure=s.procedure.value,
            start=s.appointment_time.value,
            language=session.language,
            idempotency_key=session.idempotency_key or f"call-{session.call_id}",
        )

        if not result.ok:
            alts = ", ".join(f"{a:%A %d %B at %I:%M %p}" for a in result.alternatives[:2])
            s.appointment_time.value = None
            s.appointment_time.confirmed = False
            s.appointment_time.confidence = 0.0
            session.state = "collect"
            if not result.alternatives:
                return self._escalate(session, f"no availability: {result.reason}")
            return session.say(
                f"I'm sorry, {result.reason}. I could offer {alts}. Would either work?",
                "booking rejected, offering alternatives",
            )

        session.booking_id = result.booking.booking_id
        session.state = "notify"
        if not result.replayed:
            self._queue_messages(session, result.booking, now)
        session.state = "ended"
        # No claim about the confirmation message. "Sent" can mean a mock
        # recorded it, Telegram accepted it, or nothing at all on a retry -
        # none of which is "on your WhatsApp".
        return session.say(
            f"You're booked for a {result.booking.procedure.replace('_', ' ')} on "
            f"{result.booking.start:%A %d %B at %I:%M %p}. Goodbye.",
            f"booking {result.booking.booking_id}",
        )

    def _queue_messages(self, session: CallSession, booking, now: datetime) -> None:
        params = {
            "name": booking.patient_name,
            "procedure": booking.procedure.replace("_", " "),
            "clinic": self.clinic_name,
            "when": f"{booking.start:%A %d %B at %I:%M %p}",
            "link": self.review_link,
        }
        # The reminder body already says "tomorrow" - in every language - so it
        # takes the time alone. Sharing one parameter dict across all three
        # produced "your cleaning is tomorrow at Tuesday 28 July at 11:00 AM",
        # which read as obviously wrong in English and which I could not have
        # spotted in the other four.
        reminder_params = {**params, "when": f"{booking.start:%I:%M %p}"}

        plan = [
            ("appointment_confirmation", now, params),
            ("appointment_reminder", booking.start - timedelta(days=1), reminder_params),
            ("review_request", booking.start + timedelta(hours=2), params),
        ]
        for template, when, body_params in plan:
            session.messages.append(OutboundMessage(
                template=template, to=booking.phone, language=session.language,
                parameters=body_params, send_after=when, booking_id=booking.booking_id,
            ))

        # The confirmation goes out now, through the dispatcher rather than a
        # send of its own. Marking it "sent" on any return recorded a provider
        # refusal as delivered and never retried it. The dispatcher already
        # knows ok from retryable from rejected, and never raises - the chair
        # is reserved, and a WhatsApp failure must not unwind the booking or
        # tell a caller "no" after telling them "yes".
        dispatch_due(session.messages[:1], self.messaging, self.calendar, now)

    def _escalate(self, session: CallSession, reason: str) -> str:
        session.state = "escalated"
        session.escalation_reason = reason
        # No transfer and no callback are implemented, so neither is promised.
        return session.say(
            "I'm sorry, I can't complete this booking myself. "
            "Please contact our reception team directly.",
            f"escalated: {reason}",
        )
