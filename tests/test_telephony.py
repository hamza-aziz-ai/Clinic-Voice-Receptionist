"""Bolna webhook parsing, verification and post-call ingest."""
from datetime import datetime, timedelta

import pytest

from receptionist.messaging.base import MockWhatsApp
from receptionist.scheduling.calendar import Calendar, ClinicHours
from receptionist.telephony.bolna import (
    BOLNA_WEBHOOK_SOURCE_IPS,
    BolnaWebhookError,
    confidence_ceiling,
    parse_execution,
    parse_transcript,
    verify_source,
)
from receptionist.telephony.ingest import ingest_execution
from receptionist.workflow.call import CallHandler

NOW = datetime(2026, 7, 27, 10, 0)          # a Monday; clinic open
BOLNA_IP = BOLNA_WEBHOOK_SOURCE_IPS[0]


def make_handler() -> CallHandler:
    return CallHandler(Calendar(hours=ClinicHours(), chairs=2), MockWhatsApp())


def payload(transcript: str, **over) -> dict:
    body = {
        "id": "b7140255-af33-4608-8e97-04dd944b8e48",
        "agent_id": "5bc97541-e320-4d95-a3a5-242cfe45621d",
        "status": "completed",
        "transcript": transcript,
        "conversation_duration": 42,
        "telephony_data": {
            "from_number": "+971501234567",
            "to_number": "+97143334444",
            "call_type": "inbound",
            "provider": "plivo",
            "hangup_reason": "caller_hangup",
        },
        "extracted_data": {},
        "created_at": "2026-07-27T10:00:00Z",
    }
    body.update(over)
    return body


# ---------------------------------------------------------------- transcript
def test_transcript_splits_on_speaker_not_on_newline():
    """A recited phone number wraps lines; splitting on \\n would halve it."""
    turns = parse_transcript(
        "assistant: What is your number?\n"
        "user: nine seven one five zero\none two three four five six seven\n"
        "assistant: Thank you."
    )
    assert [t.speaker for t in turns] == ["assistant", "user", "assistant"]
    assert "one two three four five six seven" in turns[1].text
    assert turns[1].text.count("nine seven one five zero") == 1


def test_transcript_without_speaker_prefixes_yields_nothing():
    """Refuse to guess. Attributing agent speech to the caller feeds our own
    prompts back into extraction."""
    assert parse_transcript("I would like a cleaning tomorrow at 3pm") == []
    assert parse_transcript("") == []


def test_caller_turns_exclude_the_agent():
    ex = parse_execution(payload(
        "assistant: Thank you for calling.\nuser: I need a cleaning.\n"
        "assistant: Certainly.\nuser: Tomorrow please."
    ))
    assert ex.caller_turns == ["I need a cleaning.", "Tomorrow please."]


# ---------------------------------------------------------------- ceilings
def test_worst_confidence_label_caps_the_call():
    ceiling = confidence_ceiling({
        "General": {"Summary": {"confidence_label": "High"}},
        "Booking": {"Phone": {"confidence_label": "Low"}},
    })
    assert ceiling == 0.50


def test_absent_or_unknown_labels_do_not_cap():
    assert confidence_ceiling({}) == 1.0
    assert confidence_ceiling({"G": {"S": {"confidence_label": "banana"}}}) == 1.0
    assert confidence_ceiling({"G": "not a dict"}) == 1.0


def test_ceiling_can_only_lower_never_raise():
    """A 'High' label must not lift a slot over a threshold it failed."""
    handler = make_handler()
    ex = parse_execution(payload(
        "assistant: Hello.\n"
        "user: my name is Priya Menon I need a cleaning tomorrow at 3 pm "
        "my number is 0501234567",
        extracted_data={"G": {"S": {"confidence_label": "High"}}},
    ))
    result = ingest_execution(ex, handler, now=NOW)
    # No word confidences and no in-call read-back: the phone slot is still
    # below threshold despite Bolna reporting high confidence.
    assert result.outcome == "needs_callback"
    assert "phone" in result.unresolved


# ---------------------------------------------------------------- verification
def test_missing_secret_fails_closed():
    with pytest.raises(BolnaWebhookError, match="no webhook secret configured"):
        verify_source(BOLNA_IP, "anything", expected_secret="")


def test_wrong_source_ip_rejected():
    with pytest.raises(BolnaWebhookError, match="not an allowed"):
        verify_source("203.0.113.9", "s3cret", "s3cret")


def test_wrong_secret_rejected_even_from_the_right_ip():
    with pytest.raises(BolnaWebhookError, match="secret mismatch"):
        verify_source(BOLNA_IP, "guess", "s3cret")


def test_both_checks_passing_is_silent():
    assert verify_source(BOLNA_IP, "s3cret", "s3cret") is None


def test_extra_allowed_ips_are_additive():
    assert verify_source("127.0.0.1", "s", "s", allowed_ips=BOLNA_WEBHOOK_SOURCE_IPS + ("127.0.0.1",)) is None


# ---------------------------------------------------------------- parsing
def test_parse_execution_maps_telephony_fields():
    ex = parse_execution(payload("user: hello"))
    assert ex.from_number == "+971501234567"
    assert ex.call_type == "inbound"
    assert ex.hangup_reason == "caller_hangup"
    assert ex.completed


def test_payload_without_an_id_is_rejected():
    body = payload("user: hello")
    del body["id"]
    with pytest.raises(BolnaWebhookError, match="no execution id"):
        parse_execution(body)


def test_unparseable_timestamp_does_not_sink_the_payload():
    ex = parse_execution(payload("user: hello", created_at="not-a-date"))
    assert ex.created_at is None


def test_utc_timestamp_becomes_clinic_local_wall_time():
    """21:00 in Dubai is 17:00Z. Left in UTC it is still the same day, but
    a 23:00 call would resolve 'tomorrow' a day early."""
    ex = parse_execution(payload("user: hello", created_at="2026-07-27T17:00:00Z"),
                         clinic_utc_offset_hours=4)
    assert ex.created_at == datetime(2026, 7, 27, 21, 0)


def test_an_already_local_offset_is_not_double_counted():
    ex = parse_execution(payload("user: hello", created_at="2026-07-27T21:00:00+04:00"),
                         clinic_utc_offset_hours=4)
    assert ex.created_at == datetime(2026, 7, 27, 21, 0)


def test_call_time_is_the_reference_for_relative_dates():
    """A retry landing the next morning must not shift the appointment."""
    handler = make_handler()
    transcript = (
        "assistant: Hi.\n"
        "user: my name is Priya Menon a cleaning tomorrow at 3 pm\n"
        "assistant: I have your name as Priya Menon. Did I get that right?\n"
        "user: yes that's right\n"
        "user: my number is 0501234567\n"
        "assistant: Is that correct?\n"
        "user: yes correct\n"
        "assistant: Shall I book that?\n"
        "user: yes please\n"
    )
    ex = parse_execution(payload(transcript, created_at="2026-07-27T06:00:00Z"))
    result = ingest_execution(ex, handler)          # no explicit now
    assert result.outcome == "booked", result.reason
    assert handler.calendar.get(result.booking_id).start == datetime(2026, 7, 28, 15, 0)


# ---------------------------------------------------------------- ingest
def test_incomplete_call_is_not_actionable():
    for status in ("queued", "ringing", "busy", "failed", "no-answer"):
        ex = parse_execution(payload("user: book me in", status=status))
        result = ingest_execution(ex, make_handler(), now=NOW)
        assert result.outcome == "not_actionable", status
        assert result.booking_id is None


def test_call_with_no_caller_speech_is_not_actionable():
    ex = parse_execution(payload("assistant: Thank you for calling. Hello?"))
    assert ingest_execution(ex, make_handler(), now=NOW).outcome == "not_actionable"


def test_in_call_readback_confirmation_books():
    """The read-back happened live; the caller's yes is in the transcript."""
    handler = make_handler()
    ex = parse_execution(payload(
        "assistant: Thank you for calling.\n"
        "user: my name is Priya Menon I need a cleaning tomorrow at 3 pm\n"
        "assistant: I have your name as Priya Menon. Did I get that right?\n"
        "user: yes that's right\n"
        "user: my number is 0501234567\n"
        "assistant: Let me confirm your number. Is that correct?\n"
        "user: yes correct\n"
        "assistant: Shall I book that?\n"
        "user: yes please\n"
    ))
    result = ingest_execution(ex, handler, now=NOW)
    assert result.outcome == "booked", result.reason
    assert result.booking_id
    assert handler.calendar.get(result.booking_id).patient_name == "Priya Menon"


def test_without_a_readback_nothing_books():
    """Invariant: no confirmation, no appointment - however complete it looks."""
    handler = make_handler()
    ex = parse_execution(payload(
        "assistant: Thank you for calling.\n"
        "user: my name is Priya Menon I need a cleaning tomorrow at 3 pm "
        "my number is 0501234567\n"
    ))
    result = ingest_execution(ex, handler, now=NOW)
    assert result.outcome == "needs_callback"
    assert result.booking_id is None
    assert handler.calendar.active() == []
    assert "phone" in result.unresolved


def test_callback_reason_names_the_specific_slots():
    """Whoever works the queue should know what to ask, not re-listen."""
    ex = parse_execution(payload("assistant: Hi.\nuser: I need a cleaning"))
    result = ingest_execution(ex, make_handler(), now=NOW)
    assert "never captured" in result.reason
    assert "phone" in result.reason


def test_redelivery_books_once_and_messages_once():
    """Bolna retries on non-2xx. Two deliveries must not mean two bookings
    and must not mean two confirmation messages to the patient."""
    handler = make_handler()
    messaging = handler.messaging
    body = payload(
        "assistant: Thank you for calling.\n"
        "user: my name is Priya Menon I need a cleaning tomorrow at 3 pm\n"
        "assistant: I have your name as Priya Menon. Did I get that right?\n"
        "user: yes that's right\n"
        "user: my number is 0501234567\n"
        "assistant: Is that correct?\n"
        "user: yes correct\n"
        "assistant: Shall I book that?\n"
        "user: yes please\n"
    )

    first = ingest_execution(parse_execution(body), handler, now=NOW)
    second = ingest_execution(parse_execution(body), handler, now=NOW)

    assert first.booking_id == second.booking_id
    assert len(handler.calendar.active()) == 1
    assert len(messaging.sent) == 1
    assert len(second.session.messages) == 0


def test_low_label_pushes_a_borderline_call_to_callback():
    """Poor audio caps every slot, which is what stops the booking."""
    handler = make_handler()
    transcript = (
        "assistant: Thank you for calling.\n"
        "user: my name is Priya Menon I need a cleaning tomorrow at 3 pm\n"
        "assistant: I have your name as Priya Menon. Did I get that right?\n"
        "user: yes that's right\n"
        "user: my number is 0501234567\n"
    )
    ex = parse_execution(payload(
        transcript, extracted_data={"G": {"Phone": {"confidence_label": "Low"}}},
    ))
    result = ingest_execution(ex, handler, now=NOW)
    assert result.outcome == "needs_callback"
    assert "audio quality capped confidence at 0.50" in result.reason


def test_a_confirmed_slot_survives_the_ceiling():
    """The caller said it aloud and agreed. That beats a statistical label."""
    handler = make_handler()
    ex = parse_execution(payload(
        "assistant: Hi.\n"
        "user: my name is Priya Menon\n"
        "assistant: I have your name as Priya Menon. Did I get that right?\n"
        "user: yes that's right\n"
        "user: a cleaning tomorrow at 3 pm\n",
        extracted_data={"G": {"S": {"confidence_label": "Low"}}},
    ))
    result = ingest_execution(ex, handler, now=NOW)
    name = result.session.slots.patient_name
    assert name.confirmed
    assert name.usable


def test_past_appointment_time_never_books_silently():
    handler = make_handler()
    ex = parse_execution(payload(
        "assistant: Hi.\n"
        "user: my name is Priya Menon a cleaning yesterday at 3 pm "
        "my number is 0501234567\n"
    ))
    result = ingest_execution(ex, handler, now=NOW + timedelta(days=2))
    assert result.outcome != "booked"
    assert handler.calendar.active() == []


def test_crosscheck_is_asked_once_per_call_not_once_per_turn():
    """The model call is the slowest thing in the webhook - 4.8s measured.
    Asking per replayed turn spends five times that on text that is not
    changing between turns."""
    calls = []

    def hook(text, now, name, phone):
        calls.append(text)
        return None

    handler = CallHandler(Calendar(hours=ClinicHours(), chairs=2),
                          MockWhatsApp(), crosscheck=hook)
    ex = parse_execution(payload(
        "assistant: Hi.\n"
        "user: my name is Priya Menon\n"
        "user: a cleaning please\n"
        "user: tomorrow at 3 pm\n"
        "user: my number is 0501234567\n"
    ))
    ingest_execution(ex, handler, now=NOW)
    assert len(calls) == 1, f"model called {len(calls)} times"
    # It is asked about the whole call, not one turn of it.
    assert "Priya Menon" in calls[0] and "0501234567" in calls[0]


def test_a_failing_crosscheck_never_fails_the_webhook():
    def exploding(text, now, name, phone):
        raise TimeoutError("ollama unreachable")

    handler = CallHandler(Calendar(hours=ClinicHours(), chairs=2),
                          MockWhatsApp(), crosscheck=exploding)
    ex = parse_execution(payload(
        "assistant: Thank you for calling.\n"
        "user: my name is Priya Menon I need a cleaning tomorrow at 3 pm\n"
        "assistant: I have your name as Priya Menon. Did I get that right?\n"
        "user: yes that's right\n"
        "user: my number is 0501234567\n"
        "assistant: Is that correct?\n"
        "user: yes correct\n"
        "assistant: Shall I book that?\n"
        "user: yes please\n"
    ))
    result = ingest_execution(ex, handler, now=NOW)
    assert result.outcome == "booked", result.reason


# A call carrying exactly the read-backs the rule extractor asks for and no
# more: without word confidences only the phone falls below its threshold, so
# there is one spare "yes". Any additional read-back therefore goes unanswered.
_EXACTLY_ENOUGH = (
    "assistant: Thank you for calling.\n"
    "user: my name is Priya Menon I need a cleaning tomorrow at 3 pm "
    "my number is 0501234567\n"
    "assistant: Let me confirm your number. Is that correct?\n"
    "user: yes correct\n"
)


def test_without_the_crosscheck_this_call_books():
    """Baseline for the test below - the difference between them is the only
    thing the second extractor changed."""
    handler = make_handler()
    result = ingest_execution(parse_execution(payload(_EXACTLY_ENOUGH)), handler, now=NOW)
    assert result.outcome == "booked", result.reason


def test_a_disagreeing_crosscheck_blocks_the_booking():
    """The whole point: the rules were confident, the second extractor read
    something else, and now a human gets asked instead of a chair being
    reserved for a treatment nobody requested."""
    from receptionist.nlu.llm_extractor import LLMExtraction

    def dissenting(text, now, name, phone):
        return LLMExtraction(procedure="extraction")

    handler = CallHandler(Calendar(hours=ClinicHours(), chairs=2),
                          MockWhatsApp(), crosscheck=dissenting)
    result = ingest_execution(parse_execution(payload(_EXACTLY_ENOUGH)),
                              handler, now=NOW)
    assert result.outcome == "needs_callback"
    assert "procedure" in result.unresolved
    assert handler.calendar.active() == []


def test_an_agreeing_crosscheck_changes_nothing():
    """Agreement must not make a borderline call book that would not have."""
    from receptionist.nlu.llm_extractor import LLMExtraction

    def agreeing(text, now, name, phone):
        return LLMExtraction(procedure="cleaning")

    handler = CallHandler(Calendar(hours=ClinicHours(), chairs=2),
                          MockWhatsApp(), crosscheck=agreeing)
    result = ingest_execution(parse_execution(payload(_EXACTLY_ENOUGH)),
                              handler, now=NOW)
    assert result.outcome == "booked", result.reason
