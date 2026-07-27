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
) -> IngestResult:
    """Replay a completed execution through the booking gate."""
    now = now or datetime.now()

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

    for turn in caller_turns:
        if session.state in ("ended", "escalated"):
            break
        # No word confidences: Bolna's transcript is a plain string. The slot
        # layer falls back to its no-metadata default, which is the honest
        # value - we genuinely do not know how well each word was heard.
        handler.handle_utterance(session, turn, now, word_confidences=None)

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
