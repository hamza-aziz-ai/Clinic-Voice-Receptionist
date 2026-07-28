from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
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


# ---------------------------------------------------------------- real call
# Every test below comes from one transcript of an actual session with the
# console, where the agent asked "What would you like to come in for?" four
# times in a row at a caller who twice said they did not understand.
NOW_REAL = datetime(2026, 7, 27, 10, 0)


def _handler():
    from receptionist.messaging.base import MockWhatsApp
    return CallHandler(Calendar(hours=ClinicHours(), chairs=2), MockWhatsApp())


def _run(handler, session, turns):
    return [handler.handle_utterance(session, t, NOW_REAL) for t in turns]


def test_a_described_symptom_is_understood_as_a_checkup():
    """Callers describe what hurts; they do not name procedures."""
    from receptionist.nlu.slots import extract_slots
    slots = extract_slots(
        "I think my wisdom tooth is not coming up properly and its aching "
        "my left side of the jaw down to the neck.", NOW_REAL)
    assert slots.procedure.value == "checkup"
    assert slots.procedure.source == "symptom"


def test_a_symptom_never_infers_a_treatment():
    """No amount of keyword matching distinguishes a wisdom tooth that needs
    removing from one that needs an X-ray. Booking an extraction off "aching"
    would be the system inventing a clinical decision from a phone call."""
    from receptionist.nlu.slots import SYMPTOM_TERMS, extract_slots
    for term in SYMPTOM_TERMS:
        slots = extract_slots(f"my tooth is {term} a lot", NOW_REAL)
        assert slots.procedure.value in (None, "checkup"), term


def test_an_inferred_procedure_is_always_read_back():
    """The caller said what hurts. They did not say what appointment they
    wanted, so it cannot book without being asked."""
    from receptionist.nlu.slots import extract_slots
    slots = extract_slots("my wisdom tooth is aching", NOW_REAL)
    assert slots.procedure.needs_confirmation
    assert not slots.procedure.usable


def test_the_readback_for_an_inferred_procedure_does_not_put_words_in_their_mouth():
    from receptionist.nlu.slots import extract_slots, readback_prompt
    slots = extract_slots("my wisdom tooth is aching", NOW_REAL)
    prompt = readback_prompt(slots.procedure)
    assert "take a look" in prompt and "check-up" in prompt
    assert "You'd like" not in prompt


def test_a_phone_number_is_read_back_in_groups():
    """Twelve digits in one run are unverifiable by ear, so the caller says
    yes because they lost track - the read-back becomes theatre."""
    from receptionist.nlu.slots import spoken_number
    assert spoken_number("+918447644188") == "plus 91 844 764 4188"
    assert spoken_number("+971501234567") == "plus 971 501 234 567"


def test_the_same_question_is_never_asked_twice():
    handler = _handler()
    session = handler.start("+918447644188")
    replies = _run(handler, session, [
        "Hi, my name is Hamza Aziz.", "+91 8447644188", "Yes, that is correct.",
        "Saturday morning works fine for me", "10:30 please",
        "Are you asking about the procedure?",
        "I don't understand what you are trying to ask.",
    ])
    asks = [r for r in replies if "?" in r]
    assert len(asks) == len(set(asks)), f"a question was repeated verbatim: {asks}"


def test_confusion_is_acknowledged_rather_than_answered_with_a_repeat():
    handler = _handler()
    session = handler.start()
    _run(handler, session, ["my name is Hamza Aziz", "0501234567", "yes",
                            "Saturday morning", "10:30"])
    reply = handler.handle_utterance(
        session, "I don't understand what you are trying to ask.", NOW_REAL)
    assert reply.startswith("Sorry, let me put that differently")


def test_a_slot_that_keeps_failing_reaches_a_human():
    handler = _handler()
    session = handler.start()
    _run(handler, session, [
        "Hi, my name is Hamza Aziz.", "+91 8447644188", "Yes, that is correct.",
        "Saturday morning works fine for me", "10:30 please",
        "Are you asking about the procedure?",
        "I don't understand what you are trying to ask.",
        "Sorry, still not sure what you mean.",
    ])
    assert session.state == "escalated"
    assert "procedure" in session.escalation_reason
    assert handler.calendar.active() == []


def test_a_vague_time_of_day_does_not_become_an_appointment():
    """"Saturday morning" is a day, not a time. The old code filled the slot
    with 10:00 and read it back as "That's Saturday 01 August at 10:00 AM.
    Shall I book that?" - proposing an hour the caller never said, where a yes
    reserves a real chair at a time nobody chose."""
    from receptionist.nlu.slots import extract_slots
    slots = extract_slots("Saturday morning works fine for me", NOW_REAL)
    assert slots.appointment_time.value is None
    assert not slots.appointment_time.usable
    assert slots.appointment_time.pending_date == date(2026, 8, 1)


def test_the_day_is_not_thrown_away_with_the_vague_time():
    """Asking "what day and time would suit you?" again, straight after the
    caller said Saturday, reads as not having listened."""
    handler = _handler()
    session = handler.start()
    _run(handler, session, ["my name is Hamza Aziz", "0501234567", "yes"])
    reply = handler.handle_utterance(session, "Saturday morning works", NOW_REAL)
    assert "Saturday 01 August" in reply
    assert "What day" not in reply


def test_a_bare_time_reply_completes_the_pending_day():
    """The answer to "what time on Saturday?" carries no day token, so the
    span regex cannot see it on its own."""
    from receptionist.nlu.slots import extract_slots
    slots = extract_slots("Saturday morning works", NOW_REAL)
    slots = extract_slots("10:30 please", NOW_REAL, existing=slots)
    assert slots.appointment_time.value == datetime(2026, 8, 1, 10, 30)
    assert slots.appointment_time.usable


def test_a_stated_time_still_books_without_a_second_question():
    """The change must not add a question for callers who were specific."""
    from receptionist.nlu.slots import extract_slots
    slots = extract_slots("Saturday at 2 pm", NOW_REAL)
    assert slots.appointment_time.value == datetime(2026, 8, 1, 14, 0)
    assert slots.appointment_time.usable


def test_the_whole_call_books_at_the_time_the_caller_chose():
    handler = _handler()
    session = handler.start("+918447644188")
    _run(handler, session, [
        "Hi, my name is Hamza Aziz.", "+91 8447644188", "Yes, that is correct.",
        "Saturday morning works fine for me", "10:30 please",
        "I think my wisdom tooth is not coming up properly and its aching "
        "my left side of the jaw down to the neck.",
        "yes please",
    ])
    assert session.state == "ended"
    booking = handler.calendar.get(session.booking_id)
    assert booking.start == datetime(2026, 8, 1, 10, 30)
    assert booking.procedure == "checkup"
    assert booking.patient_name == "Hamza Aziz"


def test_every_spoken_hour_carries_its_half_of_the_day():
    """The agent said "We're open from 9 in the morning until 8" - which is
    8 pm and reads as 8 am. An agent unclear about opening hours gets people
    turning up when the clinic is shut."""
    from datetime import time as _time
    from receptionist.nlu.normalize import spoken_time
    assert spoken_time(_time(9, 0)) == "9 in the morning"
    assert spoken_time(_time(20, 0)) == "8 in the evening"
    assert spoken_time(_time(13, 0)) == "1 in the afternoon"
    assert spoken_time(_time(12, 0)) == "12 noon"
    assert spoken_time(_time(10, 30)) == "10:30 in the morning"
    assert spoken_time(_time(0, 0)) == "12 in the morning"


def test_the_opening_hours_prompt_is_unambiguous_and_not_hardcoded():
    """Written into the sentence, the hours were a second copy of a fact the
    scheduler owns - change the closing time and the agent quotes the old one
    while the calendar refuses the booking."""
    from datetime import time as _time
    from receptionist.messaging.base import MockWhatsApp
    hours = ClinicHours(open_time=_time(8, 0), close_time=_time(17, 30))
    handler = CallHandler(Calendar(hours=hours, chairs=1), MockWhatsApp())
    prompt = handler._ask_time_on(date(2026, 8, 1), attempt=2)
    assert "8 in the morning" in prompt
    assert "5:30 in the evening" in prompt
    assert "until 8." not in prompt


def test_rejection_reasons_are_spoken_english_not_log_lines():
    """These strings go straight to the caller through the call flow."""
    from receptionist.messaging.base import MockWhatsApp
    calendar = Calendar(hours=ClinicHours(), chairs=1)
    handler = CallHandler(calendar, MockWhatsApp())

    too_early = calendar.book("A", "+971501234567", "cleaning",
                              datetime(2026, 7, 28, 7, 0))
    assert too_early.reason == "the clinic opens at 9 in the morning"

    calendar.book("B", "+971501234567", "cleaning", datetime(2026, 7, 28, 15, 0))
    full = calendar.book("C", "+971501234567", "cleaning",
                         datetime(2026, 7, 28, 15, 0))
    assert full.reason == "we're fully booked at 3 in the afternoon"
    assert "chair(s)" not in full.reason

    lunch = calendar.book("D", "+971501234567", "cleaning",
                          datetime(2026, 7, 28, 13, 15))
    assert "1 in the afternoon" in lunch.reason and "2 in the afternoon" in lunch.reason


def test_no_spoken_string_leaves_an_hour_bare():
    """Any hour the agent says aloud must carry am/pm or a part of the day."""
    import re
    from receptionist.messaging.base import MockWhatsApp
    handler = CallHandler(Calendar(hours=ClinicHours(), chairs=1), MockWhatsApp())

    spoken = [handler._ask_time_on(date(2026, 8, 1), n) for n in (1, 2, 3)]
    spoken += [p for prompts in handler.ASK_PROMPTS.values() for p in prompts]

    # Calendar dates are stripped first: "Saturday 01 August" is a date, and
    # the 01 in it is not an hour anyone could mistake for one.
    date_label = re.compile(
        r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\b")
    # (?![:\d]) stops the engine backtracking to match just the "10" of
    # "10:30" and reporting a qualified time as unqualified.
    bare = re.compile(
        r"\b\d{1,2}(?::\d{2})?(?![:\d])"
        r"(?!\s*(?:am|pm|noon|in the (?:morning|afternoon|evening)))",
        re.IGNORECASE)
    offenders = [s for s in spoken if bare.search(date_label.sub("", s))]
    assert not offenders, f"hour spoken without am/pm: {offenders}"


# ------------------------------------------------- second real transcript
# A call that failed four different ways: the agent asked for a name three
# times while the caller kept giving it, offered a time on a day the clinic
# is shut, then escalated someone who was steadily making progress.
TUE = datetime(2026, 7, 28, 10, 0)


def test_a_bare_name_answers_the_question_that_asked_for_it():
    """"Amna Ansari" has no "my name is" in front of it, so it matched
    nothing - and the agent asked again, and again, and escalated a caller
    who had answered correctly twice."""
    from receptionist.nlu.slots import extract_for_slot
    slot = extract_for_slot("Amna Ansari", "patient_name", TUE)
    assert slot.value == "Amna Ansari"
    # Inferred from context, not announced - so it is read back, never booked
    # on silently.
    assert slot.needs_confirmation


def test_a_bare_answer_is_only_used_for_the_slot_that_was_asked():
    handler = _handler()
    session = handler.start()
    reply = handler.handle_utterance(session, "Hi there", TUE)
    assert "name" in reply.lower()
    assert session.awaiting == "patient_name"

    handler.handle_utterance(session, "Amna Ansari", TUE)
    assert session.slots.patient_name.value == "Amna Ansari"


def test_a_yes_is_never_read_as_a_patient_name():
    from receptionist.nlu.slots import extract_for_slot
    for reply in ("yes", "Yes", "ok", "sure", "thanks", "hello"):
        assert extract_for_slot(reply, "patient_name", TUE) is None


def test_a_date_reply_is_not_read_as_a_patient_name():
    """A patient named "Friday Evening" is less likely than a caller
    answering the wrong question."""
    from receptionist.nlu.slots import extract_for_slot
    assert extract_for_slot("Friday evening", "patient_name", TUE) is None


def test_an_explicit_statement_still_beats_the_context_guess():
    handler = _handler()
    session = handler.start()
    handler.handle_utterance(session, "Hi there", TUE)
    handler.handle_utterance(session, "my name is Priya Menon", TUE)
    assert session.slots.patient_name.value == "Priya Menon"
    assert session.slots.patient_name.source == "asr"


def test_a_closed_day_is_refused_before_asking_what_time_suits():
    """The agent said "we're open 9 until 8, what time on Friday works?",
    took 6 pm, then refused because the clinic is closed on Fridays. It had
    the opening hours the whole time."""
    handler = _handler()
    session = handler.start()
    _run(handler, session, ["my name is Amna Ansari", "yes", "0501234567", "yes"])
    reply = handler.handle_utterance(session, "Friday evening", TUE)
    assert "closed on Friday" in reply
    assert "What time on Friday" not in reply
    assert handler.calendar.active() == []


def test_a_closed_day_with_an_hour_on_it_is_refused_just_as_early():
    """The check read only `pending_date`, so it fired for "Friday morning"
    and stayed silent for "Friday at 10 AM". The caller who gave a complete
    answer was the one who got no warning: the agent took the time, went on to
    ask what they were coming in for, and only then refused the day.

    Verbatim from a real session. The day being unbookable has nothing to do
    with whether an hour arrived with it.
    """
    handler = _handler()
    session = handler.start()
    _run(handler, session, ["my name is Amna Ansari", "yes", "0501234567", "yes"])
    reply = handler.handle_utterance(session, "Friday at 10 AM", TUE)
    assert "closed on Friday" in reply
    assert "come in for" not in reply
    # And it is gone, not left to come round the loop again.
    assert session.slots.appointment_time.value is None
    assert handler.calendar.active() == []


def test_a_closed_day_is_never_read_back_for_confirmation():
    """Asking "Friday 31 July at 10 AM, shall I book that?" is the same bug in
    a different hat - a yes to it cannot be honoured."""
    handler = _handler()
    session = handler.start()
    _run(handler, session, ["my name is Amna Ansari", "yes", "0501234567", "yes"])
    reply = handler.handle_utterance(session, "Friday at 10 AM", TUE)
    assert "shall I book" not in reply.lower()
    assert "is that correct" not in reply.lower()


def test_the_closed_day_reply_offers_a_day_that_is_open():
    handler = _handler()
    session = handler.start()
    _run(handler, session, ["my name is Amna Ansari", "yes", "0501234567", "yes"])
    reply = handler.handle_utterance(session, "Friday evening", TUE)
    assert "Saturday" in reply or "other day" in reply
    assert "Friday 31" not in reply


def test_a_value_offered_during_a_readback_is_not_a_failed_confirmation():
    """Asked to confirm a name, the caller gave their phone number - and got
    "Sorry, was that a yes or a no?" twice before the call escalated. People
    do not stay on the agent's script."""
    handler = _handler()
    session = handler.start()
    handler.handle_utterance(session, "Hi there", TUE)
    handler.handle_utterance(session, "Amna Ansari", TUE)
    assert session.state == "confirm"

    reply = handler.handle_utterance(session, "+91 8860560626", TUE)
    assert "yes or a no" not in reply
    assert session.slots.phone.value == "+918860560626"
    assert session.readback_failures == 0
    # The name is still unconfirmed - nothing bypassed the gate.
    assert not session.slots.patient_name.confirmed


def test_a_genuinely_ambiguous_reply_is_still_a_failed_confirmation():
    handler = _handler()
    session = handler.start()
    handler.handle_utterance(session, "Hi there", TUE)
    handler.handle_utterance(session, "Amna Ansari", TUE)
    reply = handler.handle_utterance(session, "hmm maybe", TUE)
    assert "yes or a no" in reply
    assert session.readback_failures == 1


def test_progress_resets_the_escalation_budget():
    """A caller steadily supplying details must not run out of turns; one
    going in circles should."""
    handler = _handler()
    session = handler.start()
    for turn in ("Hi there", "Amna Ansari", "yes", "+91 8860560626", "yes"):
        handler.handle_utterance(session, turn, TUE)
    assert session.state != "escalated"
    assert session.turns_without_progress == 0


def test_the_whole_reported_call_now_books():
    handler = _handler()
    session = handler.start("+918860560626")
    _run(handler, session, [
        "Hi there", "Amna Ansari", "Yes that is correct", "+91 8860560626",
        "Yes that is correct", "Friday evening", "Thursday 6:00 PM",
        "Having a cavity in my molar tooth", "yes please",
    ])
    assert session.state == "ended", session.escalation_reason
    booking = handler.calendar.get(session.booking_id)
    assert booking.patient_name == "Amna Ansari"
    assert booking.procedure == "filling"
    assert booking.start == datetime(2026, 7, 30, 18, 0)   # Thursday, not Friday
    assert booking.phone == "+918860560626"
