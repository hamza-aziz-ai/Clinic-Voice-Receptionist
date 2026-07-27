"""Turn a completed Bolna execution into a booking, or into a callback.

WHY THIS REPLAYS RATHER THAN RE-EXTRACTS

The obvious implementation reads ``extracted_data`` off the payload and books
what Bolna says the caller wanted. It is also the implementation that makes
every guarantee in this repository decorative: Bolna's extraction has no
notion of a per-slot threshold, so the read-back gate, the rejected-value
rule and the escalation path would all be bypassed by the one code path that
actually books real appointments.

So the caller's turns are replayed through ``CallHandler`` - the same state
machine the console drives. The read-backs already happened live, during the
call, and the caller's answers to them are sitting in the transcript. Feeding
those turns back through the gate means an in-call confirmation confirms the
slot for exactly the reason it should, and a call where the agent never asked
ends with the slot still unconfirmed and nothing booked.

Bolna's ``extracted_data`` is not thrown away. It supplies the confidence
ceiling, which can only lower a slot. It is treated as evidence about the
audio, never as a value to book on.

The three outcomes are all valid endings. ``needs_callback`` is the one worth
staring at: the call happened, the caller wanted an appointment, and the
system is deliberately declining to create one because it is not sure what it
heard. A clinic can work that queue. It cannot work a calendar full of
plausible-looking wrong bookings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from ..nlu.slots import extract_slots
from ..workflow.call import CallHandler, CallSession
from .bolna import BolnaExecution, confidence_ceiling

Outcome = Literal["booked", "needs_callback", "escalated", "not_actionable"]


@dataclass
class IngestResult:
    execution_id: str
    outcome: Outcome
    reason: str
    session: CallSession | None = None
    booking_id: str | None = None
    unresolved: list[str] = field(default_factory=list)

    @property
    def booked(self) -> bool:
        return self.outcome == "booked"

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "outcome": self.outcome,
            "reason": self.reason,
            "booking_id": self.booking_id,
            "unresolved": self.unresolved,
            "call_id": self.session.call_id if self.session else None,
        }


def ingest_execution(
    execution: BolnaExecution,
    handler: CallHandler,
    now: datetime | None = None,
    transcriber: Any = None,
) -> IngestResult:
    """Replay a completed execution through the booking gate.

    Relative dates resolve against when the *call* happened, not when the
    webhook was processed. Bolna redelivers on failure, and a retry that
    lands after midnight would otherwise read "tomorrow at 3pm" as the day
    after the one the caller meant - a wrong booking produced by a retry
    that was supposed to be a no-op.

    An explicit ``now`` still wins, so replay tooling and tests can pin the
    reference without having to forge a timestamp into the payload.
    """
    now = now or execution.created_at or datetime.now()

    if not execution.completed:
        # A ringing, busy or failed call carries no caller intent. Acting on
        # a partial one is how a hangup during the greeting becomes an
        # appointment nobody asked for.
        return IngestResult(
            execution.execution_id, "not_actionable",
            f"execution status {execution.status!r} is not {'completed'!r}",
        )

    caller_turns = execution.caller_turns
    if not caller_turns:
        return IngestResult(
            execution.execution_id, "not_actionable",
            "transcript contains no caller speech",
        )

    session = handler.start(execution.from_number)
    # Set before the first utterance so the very first extraction is clamped.
    session.confidence_ceiling = confidence_ceiling(execution.extracted_data)
    session.idempotency_key = execution.idempotency_key()
    _precompute_crosscheck(session, handler, caller_turns, now)

    # Bolna's transcript is a plain string with no confidence data, so without
    # this the slot layer falls back to its no-metadata default on every word.
    # Re-transcribing the recording is what supplies the signal the confidence
    # model was built around. None means we genuinely do not know how well each
    # word was heard, which is the honest value and not a failure.
    word_confidences = _transcribe_caller(execution, transcriber, session)

    for turn in caller_turns:
        if session.state in ("ended", "escalated"):
            break
        handler.handle_utterance(session, turn, now, word_confidences=word_confidences)

    if session.state == "escalated":
        return IngestResult(
            execution.execution_id, "escalated",
            session.escalation_reason or "escalated during the call",
            session=session,
        )

    if session.booking_id:
        return IngestResult(
            execution.execution_id, "booked", "booked from replayed transcript",
            session=session, booking_id=session.booking_id,
        )

    unresolved = [s.name for s in session.slots.all_slots() if not s.usable]
    return IngestResult(
        execution.execution_id, "needs_callback",
        _callback_reason(session, execution),
        session=session, unresolved=unresolved,
    )


def _transcribe_caller(
    execution: BolnaExecution,
    transcriber: Any,
    session: CallSession,
) -> dict[str, float] | None:
    """Re-transcribe the call recording for per-word confidence.

    Only the caller's channel. The recording contains both parties, and the
    agent says the name and the number aloud during read-backs - clearly,
    at high confidence. Letting that supply confidence for the caller's slots
    would mean the gate scoring the agent's pronunciation of a value against
    the value itself, which is circular and defeats it on exactly the fields
    it protects. The AudioRef carries the channel and the transcriber refuses
    a mixed recording outright.

    Any failure returns None, which is the behaviour that existed before this
    was wired in: fewer confident slots, more read-backs, more callbacks.
    Degraded, never wrong.
    """
    if transcriber is None or not execution.recording_url:
        return None

    from ..asr.base import AudioRef

    try:
        transcript = transcriber.transcribe(
            AudioRef(uri=execution.recording_url),
            language_hint=session.language if session.language != "uncertain" else None,
        )
    except Exception:
        return None

    if transcript is None:
        return None

    session.transcript_notes = list(transcript.notes)
    return transcript.word_confidences() or None


def _precompute_crosscheck(
    session: CallSession,
    handler: CallHandler,
    caller_turns: list[str],
    now: datetime,
) -> None:
    """Ask the second extractor once, about the whole call.

    The live path asks per turn because it only ever has one turn. Here the
    entire transcript is already in hand, and the model call is the slowest
    thing in the request - measured at 4.8 seconds against
    gpt-oss:120b-cloud. Asking once instead of once per turn takes a
    five-turn replay from roughly 24 seconds of webhook time to five, on a
    question whose answer does not change between turns.

    Failure is silent by design: no second opinion is the documented
    behaviour when the model is unavailable, and a webhook must not fail
    because a cross-check did.
    """
    if handler.crosscheck is None:
        return

    joined = " ".join(caller_turns)
    try:
        # Spans come from the rule extractor over the same joined text, so the
        # redactor removes what is actually there rather than what a single
        # turn happened to contain.
        spans = extract_slots(joined, now)
        session.crosscheck_cache = handler.crosscheck(
            joined, now, spans.patient_name.value, spans.phone.raw_text,
        )
        session.crosscheck_cached = True
    except Exception:
        return


def _callback_reason(session: CallSession, execution: BolnaExecution) -> str:
    """Say which of the several ways this ended without a booking happened.

    Written for whoever works the callback queue. "Could not book" tells them
    to listen to the whole recording; "phone was never confirmed" tells them
    what to ask in the first ten seconds.
    """
    missing = [s.name for s in session.slots.missing]
    unconfirmed = [s.name for s in session.slots.pending_confirmation]

    parts: list[str] = []
    if missing:
        parts.append(f"never captured: {', '.join(missing)}")
    if unconfirmed:
        parts.append(f"below confidence and unconfirmed: {', '.join(unconfirmed)}")
    if execution.hangup_reason:
        parts.append(f"call ended by {execution.hangup_reason}")
    if session.confidence_ceiling < 1.0:
        parts.append(f"audio quality capped confidence at {session.confidence_ceiling:.2f}")
    return "; ".join(parts) or "call ended before the booking completed"
