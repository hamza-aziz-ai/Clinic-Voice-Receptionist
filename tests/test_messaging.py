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


def test_dispatch_is_idempotent_across_passes():
    cal, bid = booked_calendar()
    m = message("appointment_reminder", booking_id=bid,
                send_after=APPOINTMENT - timedelta(days=1))
    connector = MockAiSensy()
    when = APPOINTMENT - timedelta(hours=20)
    dispatch_due([m], connector, cal, now=when)
    dispatch_due([m], connector, cal, now=when)
    assert len(connector.sent) == 1
