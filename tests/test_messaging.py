"""AiSensy request construction and scheduled dispatch."""
from datetime import datetime, timedelta

import pytest

from receptionist.messaging.aisensy import (
    CAMPAIGNS,
    PARAM_ORDER,
    AiSensyError,
    MockAiSensy,
    build_request,
)
from receptionist.messaging.base import TEMPLATES, OutboundMessage
from receptionist.messaging.dispatch import dispatch_due, is_due, is_expired
from receptionist.scheduling.calendar import Calendar, ClinicHours

NOW = datetime(2026, 7, 27, 10, 0)
APPOINTMENT = datetime(2026, 7, 28, 15, 0)

PARAMS = {
    "name": "Priya Menon", "procedure": "cleaning", "clinic": "Al Noor Dental",
    "when": "Tuesday 28 July at 03:00 PM",
    "link": "https://g.page/r/alnoor-dental/review",
}


def message(template="appointment_confirmation", language="en", **over) -> OutboundMessage:
    kwargs = dict(template=template, to="+971501234567", language=language,
                  parameters=dict(PARAMS), booking_id="bk-1")
    kwargs.update(over)
    return OutboundMessage(**kwargs)


def booked_calendar() -> tuple[Calendar, str]:
    cal = Calendar(hours=ClinicHours(), chairs=2)
    result = cal.book("Priya Menon", "+971501234567", "cleaning", APPOINTMENT)
    assert result.ok
    return cal, result.booking.booking_id


# ------------------------------------------------------------- param order
def test_params_are_positional_in_the_declared_order():
    """Not dict order. A transposition here delivers grammatical nonsense."""
    body = build_request(message(), api_key="k")
    assert body["templateParams"] == [
        "Priya Menon", "cleaning", "Al Noor Dental", "Tuesday 28 July at 03:00 PM"
    ]


def test_param_order_survives_reordered_input():
    reversed_params = {k: PARAMS[k] for k in reversed(list(PARAMS))}
    body = build_request(message(parameters=reversed_params), api_key="k")
    assert body["templateParams"][0] == "Priya Menon"
    assert body["templateParams"][1] == "cleaning"


def test_review_request_has_its_own_ordering():
    body = build_request(message("review_request"), api_key="k")
    assert body["templateParams"] == [
        "Al Noor Dental", "Priya Menon", "https://g.page/r/alnoor-dental/review"
    ]


def test_every_template_declares_a_param_order():
    assert set(PARAM_ORDER) == set(TEMPLATES)


def test_declared_order_matches_the_placeholders_in_every_language():
    """A template body and its positional mapping cannot drift apart."""
    import re
    for template, bodies in TEMPLATES.items():
        for language, body in bodies.items():
            placeholders = set(re.findall(r"\{(\w+)\}", body))
            assert placeholders == set(PARAM_ORDER[template]), (
                f"{template}/{language} placeholders {placeholders} "
                f"!= declared {PARAM_ORDER[template]}"
            )


def test_every_template_language_pair_has_a_campaign():
    for template, bodies in TEMPLATES.items():
        for language in bodies:
            assert (template, language) in CAMPAIGNS


# ------------------------------------------------------------- refusals
def test_missing_api_key_refuses():
    with pytest.raises(AiSensyError, match="no AiSensy API key"):
        build_request(message(), api_key="")


def test_unapproved_language_refuses_rather_than_falling_back():
    with pytest.raises(AiSensyError, match="no approved campaign"):
        build_request(message(language="fr"), api_key="k")


def test_missing_parameter_refuses_before_sending():
    incomplete = {k: v for k, v in PARAMS.items() if k != "when"}
    with pytest.raises(AiSensyError, match="needs when"):
        build_request(message(parameters=incomplete), api_key="k")


def test_mock_builds_the_real_body():
    """A mock that skips build_request would never catch an ordering bug."""
    connector = MockAiSensy()
    result = connector.send(message("appointment_reminder", language="ta"))
    assert result["ok"]
    assert result["campaign"] == "clinic_appointment_reminder_ta"
    assert connector.requests[0]["templateParams"][0] == "Priya Menon"


# ------------------------------------------------------------- dispatch
def test_only_due_queued_messages_are_picked_up():
    m = message(send_after=NOW + timedelta(hours=1))
    assert not is_due(m, NOW)
    assert is_due(m, NOW + timedelta(hours=2))
    m.status = "sent"
    assert not is_due(m, NOW + timedelta(hours=2))


def test_immediate_messages_are_not_the_dispatchers_job():
    """send_after is None means it went out during the call."""
    assert not is_due(message(send_after=None), NOW)


def test_due_reminder_is_sent():
    cal, bid = booked_calendar()
    m = message("appointment_reminder", booking_id=bid,
                send_after=APPOINTMENT - timedelta(days=1))
    connector = MockAiSensy()
    report = dispatch_due([m], connector, cal, now=APPOINTMENT - timedelta(hours=20))
    assert report.sent and m.status == "sent"
    assert connector.requests[0]["campaignName"] == "clinic_appointment_reminder_en"


def test_cancelled_booking_cancels_its_reminder():
    cal, bid = booked_calendar()
    cal.cancel(bid)
    m = message("appointment_reminder", booking_id=bid,
                send_after=APPOINTMENT - timedelta(days=1))
    connector = MockAiSensy()
    report = dispatch_due([m], connector, cal, now=APPOINTMENT - timedelta(hours=20))
    assert m.status == "cancelled"
    assert report.cancelled and not report.sent
    assert connector.sent == []


def test_reminder_after_the_appointment_expires_rather_than_sending_late():
    cal, bid = booked_calendar()
    m = message("appointment_reminder", booking_id=bid,
                send_after=APPOINTMENT - timedelta(days=1))
    report = dispatch_due([m], MockAiSensy(), cal, now=APPOINTMENT + timedelta(hours=1))
    assert m.status == "expired"
    assert report.expired and not report.sent


def test_review_request_expires_after_a_week():
    cal, bid = booked_calendar()
    m = message("review_request", booking_id=bid,
                send_after=APPOINTMENT + timedelta(hours=2))
    assert not is_expired(m, cal, APPOINTMENT + timedelta(days=3))
    assert is_expired(m, cal, APPOINTMENT + timedelta(days=8))


def test_review_request_is_sent_inside_the_window():
    cal, bid = booked_calendar()
    m = message("review_request", booking_id=bid,
                send_after=APPOINTMENT + timedelta(hours=2))
    report = dispatch_due([m], MockAiSensy(), cal, now=APPOINTMENT + timedelta(hours=3))
    assert report.sent and m.status == "sent"


def test_retryable_failure_stays_queued():
    class Flaky(MockAiSensy):
        def send(self, msg):
            return {"ok": False, "retryable": True, "error": "503"}

    cal, bid = booked_calendar()
    m = message("appointment_reminder", booking_id=bid,
                send_after=APPOINTMENT - timedelta(days=1))
    report = dispatch_due([m], Flaky(), cal, now=APPOINTMENT - timedelta(hours=20))
    assert m.status == "queued"
    assert report.retrying and not report.failed


def test_rejected_template_fails_and_does_not_loop():
    class Rejecting(MockAiSensy):
        def send(self, msg):
            return {"ok": False, "retryable": False, "status_code": 400}

    cal, bid = booked_calendar()
    m = message("appointment_reminder", booking_id=bid,
                send_after=APPOINTMENT - timedelta(days=1))
    report = dispatch_due([m], Rejecting(), cal, now=APPOINTMENT - timedelta(hours=20))
    assert m.status == "failed"
    assert report.failed and not report.retrying


def test_connector_raising_is_a_permanent_failure():
    cal, bid = booked_calendar()
    m = message("appointment_reminder", language="fr", booking_id=bid,
                send_after=APPOINTMENT - timedelta(days=1))
    report = dispatch_due([m], MockAiSensy(), cal, now=APPOINTMENT - timedelta(hours=20))
    assert m.status == "failed"
    assert "AiSensyError" in m.result["error"]
    assert report.failed


def test_reminder_says_tomorrow_once_not_twice():
    """The reminder body already contains "tomorrow" in all five languages,
    so passing it the full date produced "tomorrow at Tuesday 28 July at
    11:00 AM" - visibly wrong in English, invisible to me in the others."""
    from receptionist.messaging.base import render_template
    from receptionist.scheduling.calendar import Calendar as Cal
    from receptionist.workflow.call import CallHandler

    handler = CallHandler(Cal(hours=ClinicHours(), chairs=1), MockAiSensy())
    session = handler.start("+971501234567")
    session.language = "en"
    booking = handler.calendar.book(
        "Priya Menon", "+971501234567", "cleaning", APPOINTMENT).booking
    handler._queue_messages(session, booking, APPOINTMENT - timedelta(days=2))

    reminder = next(m for m in session.messages
                    if m.template == "appointment_reminder")
    confirmation = next(m for m in session.messages
                        if m.template == "appointment_confirmation")

    assert reminder.parameters["when"] == "03:00 PM"
    body = render_template("appointment_reminder", "en", reminder.parameters)
    assert "tomorrow at 03:00 PM" in body
    assert "July" not in body

    # The confirmation is not a reminder: it still carries the full date.
    assert "28 July" in confirmation.parameters["when"]


def test_dispatch_is_idempotent_across_passes():
    cal, bid = booked_calendar()
    m = message("appointment_reminder", booking_id=bid,
                send_after=APPOINTMENT - timedelta(days=1))
    connector = MockAiSensy()
    when = APPOINTMENT - timedelta(hours=20)
    dispatch_due([m], connector, cal, now=when)
    dispatch_due([m], connector, cal, now=when)
    assert len(connector.sent) == 1


# ------------------------------------------------------- live connector (httpx2)
def test_live_connector_posts_the_campaign_body(monkeypatch):
    """The real send path had no coverage at all - only the mock was tested."""
    import httpx2
    from receptionist.messaging.aisensy import AISENSY_CAMPAIGN_PATH, AiSensyConnector

    seen = {}

    class Resp:
        status_code = 200
        text = '{"success":true}'

        def json(self):
            return {"success": True}

    def fake_post(url, json=None, timeout=None):
        seen.update(url=url, body=json, timeout=timeout)
        return Resp()

    monkeypatch.setattr(httpx2, "post", fake_post)
    result = AiSensyConnector("live-key").send(message())

    assert result["ok"] and result["status_code"] == 200
    assert seen["url"] == f"https://backend.aisensy.com{AISENSY_CAMPAIGN_PATH}"
    assert seen["body"]["apiKey"] == "live-key"
    assert seen["body"]["templateParams"][0] == "Priya Menon"


def test_transport_failure_is_reported_retryable(monkeypatch):
    import httpx2
    from receptionist.messaging.aisensy import AiSensyConnector

    def boom(url, json=None, timeout=None):
        raise httpx2.ConnectError("connection refused")

    monkeypatch.setattr(httpx2, "post", boom)
    result = AiSensyConnector("live-key").send(message())
    assert result["ok"] is False and result["retryable"] is True


def test_4xx_is_not_retryable_but_5xx_is(monkeypatch):
    """Retrying a rejected template loops forever; retrying a 500 is free."""
    import httpx2
    from receptionist.messaging.aisensy import AiSensyConnector

    class Resp:
        text = "nope"

        def __init__(self, code):
            self.status_code = code

        def json(self):
            raise ValueError("not json")

    for code, retryable in ((400, False), (429, True), (500, True), (503, True)):
        monkeypatch.setattr(httpx2, "post", lambda *a, c=code, **k: Resp(c))
        result = AiSensyConnector("live-key").send(message())
        assert result["ok"] is False
        assert result["retryable"] is retryable, code
        # A non-JSON error body must not mask the status code.
        assert result["response"]["raw"] == "nope"


# ------------------------------------------- the immediate confirmation
class Scripted:
    """Answers each send from a list, recording what it was asked to send."""

    name = "scripted"

    def __init__(self, *answers):
        self.answers, self.sent = list(answers), []

    def send(self, message):
        self.sent.append(message)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def _book_with(connector, now=NOW):
    from receptionist.workflow.call import CallHandler
    cal = Calendar(hours=ClinicHours(), chairs=2)
    handler = CallHandler(cal, connector)
    session = handler.start("+971501234567")
    booking = cal.book("Priya Menon", "+971501234567", "cleaning", APPOINTMENT).booking
    handler._queue_messages(session, booking, now)
    confirmation = next(m for m in session.messages
                        if m.template == "appointment_confirmation")
    return cal, handler, session, confirmation


def test_an_accepted_confirmation_is_sent():
    _, _, _, m = _book_with(Scripted({"ok": True}))
    assert m.status == "sent"


def test_a_retryable_refusal_is_queued_not_reported_sent():
    """Marking any return as sent recorded a refusal as delivered."""
    connector = Scripted({"ok": False, "retryable": True}, {"ok": True})
    cal, handler, session, m = _book_with(connector)
    assert m.status == "queued"
    dispatch_due(session.messages, connector, cal, now=NOW + timedelta(minutes=15))
    assert m.status == "sent"
    assert connector.sent.count(m) == 2


def test_a_permanent_refusal_or_an_exception_fails_without_unwinding_the_booking():
    for answer in ({"ok": False}, RuntimeError("provider down")):
        cal, _, _, m = _book_with(Scripted(answer))
        assert m.status == "failed"
        assert len(cal.active()) == 1


def test_a_cancelled_booking_cancels_a_confirmation_still_being_retried():
    connector = Scripted({"ok": False, "retryable": True})
    cal, _, session, m = _book_with(connector)
    cal.cancel(m.booking_id)
    dispatch_due(session.messages, connector, cal, now=NOW + timedelta(minutes=15))
    assert m.status == "cancelled"
    assert len(connector.sent) == 1


def test_a_confirmation_still_queued_at_the_appointment_expires():
    connector = Scripted({"ok": False, "retryable": True})
    cal, _, session, m = _book_with(connector)
    dispatch_due([m], connector, cal, now=APPOINTMENT)
    assert m.status == "expired"
    assert len(connector.sent) == 1


def test_overlapping_dispatch_passes_send_a_message_once():
    """The start barrier sits outside the lock: the passes start together, and
    the lock serialises them, so the second finds the message already sent."""
    import threading

    cal, bid = booked_calendar()
    m = message("appointment_reminder", booking_id=bid,
                send_after=APPOINTMENT - timedelta(days=1))
    connector = Scripted(*([{"ok": True}] * 8))
    start = threading.Barrier(8)

    def one_pass():
        start.wait()
        dispatch_due([m], connector, cal, now=APPOINTMENT - timedelta(hours=20))

    threads = [threading.Thread(target=one_pass) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert m.status == "sent"
    assert len(connector.sent) == 1


def test_an_english_fallback_is_logged_without_the_patient_details(caplog):
    import logging

    from receptionist.messaging.base import render_template

    with caplog.at_level(logging.WARNING, logger="receptionist.messaging.base"):
        render_template("appointment_confirmation", "fr", PARAMS)
    assert "'fr'" in caplog.text and "appointment_confirmation" in caplog.text
    assert "Priya" not in caplog.text and "+971" not in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="receptionist.messaging.base"):
        render_template("appointment_confirmation", "ml", PARAMS)
    assert caplog.text == ""
