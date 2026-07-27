"""The free messaging channel, and the constraint that makes it not a
drop-in replacement for WhatsApp."""
from datetime import datetime, timedelta

import pytest

from receptionist.messaging.base import OutboundMessage
from receptionist.messaging.dispatch import dispatch_due
from receptionist.messaging.telegram import (
    ChatDirectory,
    MockTelegram,
    TelegramError,
    build_message,
)
from receptionist.scheduling.calendar import Calendar, ClinicHours

APPOINTMENT = datetime(2026, 7, 28, 15, 0)
PHONE = "+971501234567"

PARAMS = {
    "name": "Priya Menon", "procedure": "cleaning", "clinic": "Al Noor Dental",
    "when": "Tuesday 28 July at 03:00 PM",
    "link": "https://g.page/r/alnoor-dental/review",
}


def message(template="appointment_confirmation", language="en", **over) -> OutboundMessage:
    kwargs = dict(template=template, to=PHONE, language=language,
                  parameters=dict(PARAMS), booking_id="bk-1")
    kwargs.update(over)
    return OutboundMessage(**kwargs)


def linked(chat_id: int = 4242) -> ChatDirectory:
    d = ChatDirectory()
    d.link(PHONE, chat_id)
    return d


# ------------------------------------------------- the opt-in constraint
def test_an_unlinked_patient_cannot_be_messaged_at_all():
    """Telegram addresses a chat_id, not a phone number, and a chat_id only
    exists once the patient has messaged the bot. This is the shape of the
    product, not a permission that can be configured away."""
    with pytest.raises(TelegramError, match="has not started a chat"):
        build_message(message(), ChatDirectory())


def test_the_failure_names_the_number_so_reception_can_act():
    try:
        build_message(message(), ChatDirectory())
    except TelegramError as exc:
        assert PHONE in str(exc)


def test_numbers_are_matched_on_digits_not_formatting():
    """The calendar stores E.164 and a patient may share any format. Matching
    raw strings would look identical to never having linked."""
    directory = ChatDirectory()
    directory.link("050 123 4567", 99)
    assert directory.chat_id_for("+971501234567") is None   # different number
    directory.link("+971501234567", 4242)
    assert directory.chat_id_for("971501234567") == 4242
    assert directory.chat_id_for("+971 50 123 4567") == 4242


# ------------------------------------------------- body
def test_the_body_is_the_rendered_template_for_the_call_language():
    body = build_message(message(language="ta"), linked())
    assert body["chat_id"] == 4242
    assert "வணக்கம்" in body["text"]
    assert "Priya Menon" in body["text"]


def test_no_parse_mode_is_set():
    """Markdown would read an underscore or asterisk in a patient's name as
    formatting and either mangle the name or fail the send on unbalanced
    markup."""
    assert "parse_mode" not in build_message(message(), linked())


def test_a_name_with_markdown_characters_survives_intact():
    params = {**PARAMS, "name": "Anne-Marie *O_Brien*"}
    body = build_message(message(parameters=params), linked())
    assert "Anne-Marie *O_Brien*" in body["text"]


def test_the_review_request_arrives_quietly():
    """A confirmation should buzz the phone. A review ask should not."""
    assert build_message(message("review_request"), linked())["disable_notification"]
    assert not build_message(message(), linked())["disable_notification"]


def test_every_template_and_language_renders():
    from receptionist.messaging.base import TEMPLATES
    for template, bodies in TEMPLATES.items():
        for language in bodies:
            body = build_message(message(template, language), linked())
            assert body["text"].strip()


# ------------------------------------------------- dispatcher contract
def test_an_unlinked_patient_is_a_permanent_dispatch_failure():
    """Not retryable: the patient has to press Start, and retrying will not
    make that happen. It must surface, not vanish."""
    calendar = Calendar(hours=ClinicHours(), chairs=1)
    booking = calendar.book("Priya Menon", PHONE, "cleaning", APPOINTMENT).booking
    m = message("appointment_reminder", booking_id=booking.booking_id,
                send_after=APPOINTMENT - timedelta(days=1))

    report = dispatch_due([m], MockTelegram(), calendar,
                          now=APPOINTMENT - timedelta(hours=20))
    assert m.status == "failed"
    assert report.failed and not report.retrying
    assert "has not started a chat" in m.result["error"]


def test_a_linked_patient_is_dispatched_normally():
    calendar = Calendar(hours=ClinicHours(), chairs=1)
    booking = calendar.book("Priya Menon", PHONE, "cleaning", APPOINTMENT).booking
    m = message("appointment_reminder", booking_id=booking.booking_id,
                send_after=APPOINTMENT - timedelta(days=1))

    connector = MockTelegram(linked())
    report = dispatch_due([m], connector, calendar,
                          now=APPOINTMENT - timedelta(hours=20))
    assert report.sent and m.status == "sent"
    assert connector.requests[0]["chat_id"] == 4242


def test_the_whole_call_flow_works_on_telegram():
    """The point of the MessagingConnector interface, cashed rather than
    asserted: the call flow does not know the channel changed."""
    from receptionist.workflow.call import CallHandler

    connector = MockTelegram(linked())
    handler = CallHandler(Calendar(hours=ClinicHours(), chairs=2), connector)
    session = handler.start(PHONE)
    now = datetime(2026, 7, 27, 10, 0)
    for turn in ("my name is Priya Menon I need a cleaning tomorrow at 3 pm "
                 f"my number is {PHONE}", "yes correct"):
        handler.handle_utterance(session, turn, now)

    assert session.booking_id, "call did not reach a booking"
    assert len(connector.sent) == 1               # confirmation only, immediately
    assert "Priya Menon" in connector.requests[0]["text"]
