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

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal

from ..messaging.base import MessagingConnector, OutboundMessage, render_template
from ..nlu.language import LANGUAGE_NAMES, detect_language
from ..nlu.slots import SlotSet, extract_slots, readback_prompt
from ..scheduling.calendar import Calendar

CallState = Literal[
    "greeting", "detect_language", "collect", "confirm",
    "book", "notify", "ended", "escalated",
]

MAX_READBACK_FAILURES = 2
MAX_COLLECT_TURNS = 8


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
    messages: list[OutboundMessage] = field(default_factory=list)
    # Upper bound on any slot's confidence for this call. 1.0 means the
    # transcript is the only evidence. A replayed Bolna execution sets this
    # from the extraction confidence labels, because poor audio is a property
    # of the call rather than of one field. It can only ever lower a slot.
    confidence_ceiling: float = 1.0
    # Supplied when the call originates from a source that can redeliver, so
    # a retry reserves the same appointment instead of a second one.
    idempotency_key: str | None = None

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
    ) -> None:
        self.calendar = calendar
        self.messaging = messaging
        self.clinic_name = clinic_name
        self.review_link = review_link

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
    ) -> str:
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
            return self._handle_readback_answer(session, text, now, word_confidences)

        if session.state == "collect":
            return self._collect(session, text, now, word_confidences)

        return session.say("Sorry, could you repeat that?")

    # ------------------------------------------------------------------
    def _collect(
        self, session: CallSession, text: str, now: datetime,
        word_confidences: dict[str, float] | None,
    ) -> str:
        session.collect_turns += 1
        session.slots = extract_slots(text, now, word_confidences, session.slots)
        self._apply_ceiling(session)

        if session.collect_turns > MAX_COLLECT_TURNS:
            return self._escalate(session, "too many turns without a complete booking")

        pending = session.slots.pending_confirmation
        if pending:
            session.state = "confirm"
            session._pending_slot = pending[0]           # type: ignore[attr-defined]
            return session.say(
                readback_prompt(pending[0], session.language),
                f"{pending[0].name} confidence {pending[0].confidence:.2f} below threshold",
            )

        missing = session.slots.missing
        if missing:
            return session.say(self._ask_for(missing[0].name), f"missing {missing[0].name}")

        return self._book(session, now)

    def _handle_readback_answer(
        self, session: CallSession, text: str, now: datetime,
        word_confidences: dict[str, float] | None,
    ) -> str:
        slot = getattr(session, "_pending_slot", None)
        lowered = text.lower()
        affirmative = any(w in lowered for w in
                          ("yes", "correct", "right", "yeah", "yep", "sari", "seri", "haan", "ok"))
        negative = any(w in lowered for w in
                       ("no", "not", "wrong", "illai", "illa", "nahi", "incorrect"))

        if slot is None:
            session.state = "collect"
            return self._collect(session, text, now, word_confidences)

        if affirmative and not negative:
            slot.confirm()
            session.state = "collect"
            session.collect_turns -= 1     # a successful read-back is not a failed turn
            return self._collect(session, "", now, word_confidences)

        if negative:
            slot.reject()
            session.readback_failures += 1
            if session.readback_failures > MAX_READBACK_FAILURES:
                return self._escalate(session, "repeated failure to confirm details")
            session.state = "collect"
            return session.say(self._ask_for(slot.name), "read-back rejected, re-asking")

        # Ambiguous answer to a yes/no question - do not treat silence as yes.
        session.readback_failures += 1
        if session.readback_failures > MAX_READBACK_FAILURES:
            return self._escalate(session, "could not obtain a clear confirmation")
        return session.say("Sorry, was that a yes or a no?", "ambiguous confirmation")

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

    def _ask_for(self, slot_name: str) -> str:
        return {
            "patient_name": "Could I take your full name please?",
            "phone": "What is the best mobile number to reach you on?",
            "appointment_time": "What day and time would suit you?",
            "procedure": "What would you like to come in for?",
        }[slot_name]

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
            self._queue_messages(session, result.booking)
        session.state = "ended"
        return session.say(
            f"You're booked for a {result.booking.procedure.replace('_', ' ')} on "
            f"{result.booking.start:%A %d %B at %I:%M %p}. "
            f"I've sent a confirmation to your WhatsApp. Goodbye.",
            f"booking {result.booking.booking_id}",
        )

    def _queue_messages(self, session: CallSession, booking) -> None:
        params = {
            "name": booking.patient_name,
            "procedure": booking.procedure.replace("_", " "),
            "clinic": self.clinic_name,
            "when": f"{booking.start:%A %d %B at %I:%M %p}",
            "link": self.review_link,
        }
        plan = [
            ("appointment_confirmation", None),
            ("appointment_reminder", booking.start - timedelta(days=1)),
            ("review_request", booking.start + timedelta(hours=2)),
        ]
        for template, when in plan:
            msg = OutboundMessage(
                template=template, to=booking.phone, language=session.language,
                parameters=params, send_after=when, booking_id=booking.booking_id,
            )
            session.messages.append(msg)
            if when is None:                       # immediate
                msg.result = self.messaging.send(msg)
                msg.status = "sent"

    def _escalate(self, session: CallSession, reason: str) -> str:
        session.state = "escalated"
        session.escalation_reason = reason
        return session.say(
            "Let me put you through to our reception team who can help. One moment.",
            f"escalated: {reason}",
        )
