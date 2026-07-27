"""Bolna post-call webhook payload.

WHAT BOLNA ACTUALLY SENDS, AND WHAT IT COSTS US

Bolna POSTs the execution object after the call ends. Two properties of that
payload drove the design here, and both contradicted what I assumed before
reading the API reference:

1. ``transcript`` is a single **string** - ``"assistant: ...\\nuser: ..."`` -
   not a token stream. There are **no per-word ASR confidences**. The slot
   layer was built to average word confidence over the span a value came
   from; over this transcript that signal simply does not exist, and
   ``_asr_confidence`` falls back to its no-metadata default.

2. There is **no signature header**. The documented protection is source-IP
   allowlisting from Bolna's egress address. That is not authentication:
   anyone who learns the URL and can spoof or proxy from that address can
   post a booking. So this module requires a shared secret of our own *in
   addition* to the IP check, and refuses the payload if either is absent.

The consequence of (1) is the interesting one. Losing word confidence means
every slot sits near the no-metadata default, which is below the phone
threshold - so a post-call payload can never clear the confidence gate on
its own. That is the correct outcome, not a defect: nobody was on the line
to be asked. What rescues the booking is that the read-backs already
happened *during* the call, and the caller's answers are in the transcript.
See ``ingest.py`` - the turns are replayed through the same state machine
the live console drives, so an in-call confirmation confirms the slot and
nothing new gets a bypass around the gate.
"""
from __future__ import annotations

import hmac
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

# Documented egress address for Bolna webhook deliveries. Kept as data
# rather than hardcoded in a comparison so a deployment can extend it
# without editing the check.
BOLNA_WEBHOOK_SOURCE_IPS: tuple[str, ...] = ("13.203.39.153",)

# Only a finished conversation can be booked on. A queued, ringing, busy or
# failed execution has no caller intent to act on, and treating an errored
# call as bookable is exactly the silent failure this project exists to stop.
BOOKABLE_STATUS = "completed"

Speaker = Literal["assistant", "user"]

# Bolna's data-extraction output labels each field's reliability rather than
# giving a number. These map to ceilings, never to floors: the label can only
# ever *lower* a slot's confidence, so a confident-looking label cannot lift
# a value over the threshold it failed on its own merits.
CONFIDENCE_LABEL_CEILING: dict[str, float] = {
    "high": 1.00,
    "medium": 0.80,
    "low": 0.50,
}

_TURN = re.compile(r"^(assistant|user)\s*:\s*", re.IGNORECASE | re.MULTILINE)


class BolnaWebhookError(ValueError):
    """Payload rejected before any interpretation was attempted."""


@dataclass
class TranscriptTurn:
    speaker: Speaker
    text: str


@dataclass
class BolnaExecution:
    """The subset of the execution object this system acts on.

    Deliberately not a passthrough of every field Bolna sends. Cost
    breakdowns and provider hangup codes are useful for their dashboard and
    irrelevant to whether we may book, and copying them in invites someone
    to start branching on them.
    """

    execution_id: str
    agent_id: str
    status: str
    transcript: str
    from_number: str = ""
    to_number: str = ""
    call_type: str = "inbound"
    provider_call_id: str = ""
    recording_url: str = ""
    hangup_reason: str = ""
    conversation_duration: float = 0.0
    extracted_data: dict[str, Any] = field(default_factory=dict)
    context_details: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None

    @property
    def completed(self) -> bool:
        return self.status == BOOKABLE_STATUS

    @property
    def turns(self) -> list[TranscriptTurn]:
        return parse_transcript(self.transcript)

    @property
    def caller_turns(self) -> list[str]:
        return [t.text for t in self.turns if t.speaker == "user"]

    def idempotency_key(self) -> str:
        """Stable across redeliveries of the same call.

        Bolna retries on non-2xx, so the same execution can arrive several
        times. Keying the booking on the execution id means a retry returns
        the original booking rather than a second appointment.
        """
        return f"bolna-{self.execution_id}"


def parse_transcript(transcript: str) -> list[TranscriptTurn]:
    """Split ``"assistant: ...\\nuser: ..."`` into turns.

    A turn runs until the next speaker prefix, not until the next newline -
    a caller reciting a phone number often wraps across lines, and splitting
    on newlines would cut the digits in half.
    """
    if not transcript or not transcript.strip():
        return []

    marks = list(_TURN.finditer(transcript))
    if not marks:
        # No speaker prefixes at all. Attributing this to the caller would
        # feed the agent's own words back into extraction, so refuse to guess.
        return []

    turns: list[TranscriptTurn] = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(transcript)
        text = transcript[m.end():end].strip()
        if text:
            turns.append(TranscriptTurn(m.group(1).lower(), text))  # type: ignore[arg-type]
    return turns


def confidence_ceiling(extracted_data: dict[str, Any]) -> float:
    """Lowest confidence ceiling implied by Bolna's extraction labels.

    ``extracted_data`` is nested by category then field, each carrying a
    ``confidence_label``. The worst label on the call caps every slot: if
    Bolna was unsure about anything it heard, that is evidence about the
    audio, and audio quality is not per-field.
    """
    ceiling = 1.0
    for category in extracted_data.values():
        if not isinstance(category, dict):
            continue
        for entry in category.values():
            if not isinstance(entry, dict):
                continue
            label = str(entry.get("confidence_label", "")).strip().lower()
            if label in CONFIDENCE_LABEL_CEILING:
                ceiling = min(ceiling, CONFIDENCE_LABEL_CEILING[label])
    return ceiling


def verify_source(
    remote_ip: str,
    provided_secret: str,
    expected_secret: str,
    allowed_ips: tuple[str, ...] = BOLNA_WEBHOOK_SOURCE_IPS,
) -> None:
    """Both checks, or the payload does not get interpreted.

    IP allowlisting alone is what Bolna documents, and it is not enough to
    authenticate a request that creates appointments and sends WhatsApp
    messages to real numbers. The shared secret is ours, compared with
    ``hmac.compare_digest`` so a wrong secret cannot be recovered by timing
    the rejection.

    An unset secret is a hard failure rather than a skipped check. The
    tempting alternative - "no secret configured, so allow" - turns a missing
    environment variable into an open booking endpoint, and it fails open on
    exactly the deployment most likely to be misconfigured.
    """
    if not expected_secret:
        raise BolnaWebhookError(
            "no webhook secret configured; refusing to accept unauthenticated bookings"
        )
    if remote_ip not in allowed_ips:
        raise BolnaWebhookError(f"source {remote_ip!r} is not an allowed Bolna egress address")
    if not hmac.compare_digest(provided_secret or "", expected_secret):
        raise BolnaWebhookError("webhook secret mismatch")


def _clinic_local(raw: Any, offset_hours: float) -> datetime | None:
    """ISO-8601 timestamp to naive clinic-local time, or None if unusable.

    An unparseable timestamp yields None rather than raising. The caller
    falls back to the current time, which is a slightly worse reference for
    relative dates - not a reason to drop a call that otherwise booked.
    """
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed
    # Normalise to UTC first: the payload may carry any offset, and adding
    # the clinic offset to an already-local timestamp double-counts it.
    utc = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return utc + timedelta(hours=offset_hours)


def parse_execution(
    payload: dict[str, Any],
    clinic_utc_offset_hours: float = 4.0,
) -> BolnaExecution:
    """Map the webhook body onto the fields this system uses.

    Every Bolna-specific field name is read here and nowhere else, so a
    change to their payload is a single-file edit rather than a hunt.

    ``created_at`` arrives UTC and is converted to clinic-local wall time,
    naive, because that is what the rest of the system works in and what
    "tomorrow at 3pm" means to a caller. Keeping it UTC would resolve a
    21:00 Dubai call to the following day in the extractor, silently booking
    everyone who rings after 20:00 a day late.
    """
    if not isinstance(payload, dict):
        raise BolnaWebhookError("payload is not a JSON object")

    execution_id = payload.get("id") or payload.get("execution_id")
    if not execution_id:
        raise BolnaWebhookError("payload has no execution id")

    telephony = payload.get("telephony_data") or {}
    if not isinstance(telephony, dict):
        telephony = {}

    created = _clinic_local(payload.get("created_at"), clinic_utc_offset_hours)

    return BolnaExecution(
        execution_id=str(execution_id),
        agent_id=str(payload.get("agent_id") or ""),
        status=str(payload.get("status") or "").lower(),
        transcript=str(payload.get("transcript") or ""),
        from_number=str(telephony.get("from_number") or ""),
        to_number=str(telephony.get("to_number") or ""),
        call_type=str(telephony.get("call_type") or "inbound"),
        provider_call_id=str(telephony.get("provider_call_id") or ""),
        recording_url=str(telephony.get("recording_url") or ""),
        hangup_reason=str(telephony.get("hangup_reason") or ""),
        conversation_duration=float(payload.get("conversation_duration") or 0),
        extracted_data=payload.get("extracted_data") or {},
        context_details=payload.get("context_details") or {},
        created_at=created,
    )
