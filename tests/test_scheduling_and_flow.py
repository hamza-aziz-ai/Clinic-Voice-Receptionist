from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import pytest
from receptionist.messaging.base import MockWhatsApp, TEMPLATES, render_template
from receptionist.scheduling.calendar import Calendar, ClinicHours
from receptionist.workflow.call import CallHandler

NOW = datetime(2026, 7, 27, 9, 0)
SLOT = datetime(2026, 7, 28, 15, 0)


@pytest.fixture
def calendar():
    return Calendar(chairs=1)


class TestCalendarSafety:
    def test_concurrent_calls_cannot_double_book(self, calendar):
        """20 simultaneous calls for one chair must yield exactly one booking."""
        with ThreadPoolExecutor(max_workers=20) as ex:
            results = list(ex.map(
                lambda i: calendar.book(f"P{i}", "+971501234567", "cleaning", SLOT),
                range(20)))
        assert sum(r.ok for r in results) == 1
        assert len(calendar.active()) == 1

    def test_idempotency_key_survives_webhook_retry(self, calendar):
        a = calendar.book("Ravi", "+919876543210", "checkup", SLOT, idempotency_key="c1")
        b = calendar.book("Ravi", "+919876543210", "checkup", SLOT, idempotency_key="c1")
        assert a.booking.booking_id == b.booking.booking_id
        assert len(calendar.active()) == 1

    def test_rejection_offers_alternatives(self, calendar):
        calendar.book("A", "+971501234567", "cleaning", SLOT)
        r = calendar.book("B", "+971501234567", "cleaning", SLOT)
        assert not r.ok and r.alternatives

    def test_multiple_chairs_allow_parallel_bookings(self):
        cal = Calendar(chairs=3)
        assert sum(cal.book(f"P{i}", "+971501234567", "cleaning", SLOT).ok
                   for i in range(4)) == 3

    def test_closed_day_rejected(self, calendar):
        r = calendar.book("A", "+971501234567", "checkup", datetime(2026, 7, 31, 10, 0))
        assert not r.ok and "closed" in r.reason

    def test_lunch_break_rejected(self, calendar):
        r = calendar.book("A", "+971501234567", "root_canal", datetime(2026, 7, 28, 12, 30))
        assert not r.ok and "break" in r.reason

    def test_overrunning_closing_time_rejected(self, calendar):
        r = calendar.book("A", "+971501234567", "root_canal", datetime(2026, 7, 28, 19, 30))
        assert not r.ok

    def test_procedure_duration_drives_the_conflict_window(self, calendar):
        calendar.book("A", "+971501234567", "root_canal", SLOT)          # 90 min
        assert not calendar.book("B", "+971501234567", "checkup",
                                 datetime(2026, 7, 28, 16, 0)).ok
        assert calendar.book("C", "+971501234567", "checkup",
                             datetime(2026, 7, 28, 16, 45)).ok

    def test_cancellation_frees_the_slot(self, calendar):
        b = calendar.book("A", "+971501234567", "cleaning", SLOT)
        assert calendar.cancel(b.booking.booking_id)
        assert calendar.book("B", "+971501234567", "cleaning", SLOT).ok


class TestMessaging:
    def test_all_templates_exist_in_all_languages(self):
        for template, bodies in TEMPLATES.items():
            for lang in ("en", "ta", "kn", "ml", "hi"):
                assert lang in bodies, f"{template} missing {lang}"

    def test_fallback_to_english_for_unknown_language(self):
        body = render_template("appointment_confirmation", "fr",
                               {"name": "X", "procedure": "cleaning",
                                "clinic": "C", "when": "now", "link": "l"})
        assert "confirmed" in body


class TestCallFlow:
    def _handler(self, chairs=2):
        cal, wa = Calendar(chairs=chairs), MockWhatsApp()
        return CallHandler(cal, wa), cal, wa

    def test_happy_path_books_and_notifies(self):
        h, cal, wa = self._handler()
        s = h.start("+971501112222")
        for text in ("my name is Priya Menon I need a cleaning", "yes",
                     "my number is nine seven one five zero one two three four five six seven",
                     "yes", "tomorrow at 3 pm", "yes"):
            h.handle_utterance(s, text, NOW)
        assert s.state == "ended"
        assert s.booking_id and len(cal.active()) == 1
        assert len(s.messages) == 3            # confirmation, reminder, review
        assert len(wa.sent) == 1               # only the confirmation goes immediately

    def test_nothing_books_before_readbacks_clear(self):
        h, cal, _ = self._handler()
        s = h.start()
        h.handle_utterance(s, "my name is Priya Menon I need a cleaning tomorrow at 3 pm "
                              "my number is nine seven one five zero one two three four five six seven",
                           NOW, {"priya": 0.3, "menon": 0.3})
        assert s.state == "confirm"
        assert not cal.active()

    def test_ambiguous_answer_is_not_treated_as_yes(self):
        h, cal, _ = self._handler()
        s = h.start()
        h.handle_utterance(s, "my name is Priya Menon", NOW, {"priya": 0.3, "menon": 0.3})
        reply = h.handle_utterance(s, "hmm maybe", NOW)
        assert "yes or a no" in reply
        assert not s.slots.patient_name.confirmed

    def test_repeated_confirmation_failure_escalates(self):
        """Three failed read-backs hands off to a human rather than guessing."""
        h, _, _ = self._handler()
        s = h.start()
        weak = {"priya": 0.3, "menon": 0.3}
        for _ in range(4):
            h.handle_utterance(s, "my name is Priya Menon", NOW, weak)
            if s.state == "escalated":
                break
            h.handle_utterance(s, "no that's wrong", NOW)
        assert s.state == "escalated"
        assert s.escalation_reason

    def test_full_calendar_offers_alternatives_rather_than_failing(self):
        h, cal, _ = self._handler(chairs=1)
        cal.book("Someone", "+971501234567", "cleaning", datetime(2026, 7, 28, 15, 0))
        s = h.start()
        replies = [h.handle_utterance(s, text, NOW) for text in (
            "my name is Priya Menon I need a cleaning", "yes",
            "my number is nine seven one five zero one two three four five six seven",
            "yes", "tomorrow at 3 pm", "yes")]
        assert any("could offer" in r for r in replies)
        assert s.state == "collect"
        assert len(cal.active()) == 1        # the caller was NOT double-booked

    def test_language_is_detected_and_carried_into_messages(self):
        h, cal, wa = self._handler()
        s = h.start()
        h.handle_utterance(s, "നമസ്കാരം, my name is Anjali Nair, എനിക്ക് filling വേണം", NOW)
        assert s.language == "ml"
        for text in ("yes", "my number is nine seven one five zero one two three four five six seven",
                     "yes", "tomorrow at 4 pm", "yes"):
            h.handle_utterance(s, text, NOW)
        assert s.booking_id
        assert all(m.language == "ml" for m in s.messages)
