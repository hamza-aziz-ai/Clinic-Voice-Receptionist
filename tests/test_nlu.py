from __future__ import annotations
from datetime import datetime
import pytest
from receptionist.nlu.language import detect_language
from receptionist.nlu.normalize import normalise_datetime, normalise_name, normalise_phone
from receptionist.nlu.slots import CONFIRMATION_THRESHOLDS, extract_slots, readback_prompt

REF = datetime(2026, 7, 27, 9, 0)   # Monday


class TestLanguageDetection:
    @pytest.mark.parametrize("text,expected", [
        ("I need an appointment tomorrow", "en"),
        ("எனக்கு ஒரு நேரம் வேண்டும்", "ta"),
        ("ಸರಿ ನನಗೆ ಬೇಕು", "kn"),
        ("എനിക്ക് ഒരു അപ്പോയിന്റ്മെന്റ് വേണം", "ml"),
        ("मुझे अपॉइंटमेंट चाहिए", "hi"),
    ])
    def test_script_detection(self, text, expected):
        assert detect_language(text).language == expected

    def test_romanised_fallback(self):
        d = detect_language("namaskaram enikku appointment venam illa")
        assert d.language == "ml" and d.method == "romanised"

    def test_code_switched_call_detects_indic_not_english(self):
        """Regression: Indic combining marks are category Mn, not isalpha().

        Filtering on isalpha() first dropped ~20% of the Indic characters and
        pushed every code-switched utterance below the detection threshold.
        """
        text = ("வணக்கம், my name is Karthik Raman, எனக்கு cleaning வேண்டும் "
                "tomorrow morning, number nine seven one five zero one two")
        assert detect_language(text).language == "ta"

    def test_empty_is_uncertain_not_english(self):
        assert detect_language("").language == "uncertain"
        assert detect_language("   ").language == "uncertain"

    def test_uncertainty_is_a_valid_outcome(self):
        assert not detect_language("").is_confident


class TestPhoneNormalisation:
    @pytest.mark.parametrize("raw,expected", [
        ("nine seven one five zero one two three four five six seven", "+971501234567"),
        ("+971 50 123 4567", "+971501234567"),
        ("0501234567", "+971501234567"),
        ("nine one nine eight seven six five four three two one zero", "+919876543210"),
    ])
    def test_valid_numbers(self, raw, expected):
        assert normalise_phone(raw).e164 == expected

    def test_double_multiplier_expands(self):
        r = normalise_phone("double nine eight seven six five four three two one")
        # "double nine" -> 99, then 8 more digits: 9987654321, prefixed +91
        assert r is not None and "9987654321" in r.e164

    def test_multiplier_output_too_short_is_still_rejected(self):
        assert normalise_phone("double nine eight seven") is None

    def test_too_short_is_rejected_not_guessed(self):
        """A nearly-right number is worse than none - it reaches a stranger."""
        assert normalise_phone("12345") is None
        assert normalise_phone("") is None

    def test_unrecognised_pattern_carries_a_confidence_penalty(self):
        r = normalise_phone("one two three four five six seven eight nine zero")
        assert r is not None and r.confidence_penalty > 0


class TestDateTimeNormalisation:
    def test_relative_days(self):
        assert normalise_datetime("tomorrow at 3 pm", REF).when == datetime(2026, 7, 28, 15, 0)

    def test_day_of_month_is_not_read_as_the_hour(self):
        """Regression: '15/08 at 11am' parsed as 15:00, booking four hours late."""
        assert normalise_datetime("15/08 at 11am", REF).when == datetime(2026, 8, 15, 11, 0)

    def test_vague_time_is_flagged(self):
        r = normalise_datetime("tomorrow morning", REF)
        assert r.time_was_vague is True

    def test_bare_low_hour_assumed_afternoon(self):
        """'at 3' in a clinic means 15:00, not 03:00."""
        assert normalise_datetime("tomorrow at 3", REF).when.hour == 15

    def test_unparseable_returns_none(self):
        assert normalise_datetime("sometime soon", REF) is None

    def test_invalid_calendar_date_rejected(self):
        assert normalise_datetime("31/02 at 10am", REF) is None


class TestNameNormalisation:
    def test_strips_lead_in_phrases(self):
        assert normalise_name("my name is ravi kumar") == "Ravi Kumar"

    def test_stops_at_clause_boundary(self):
        """Regression: over-capture scored 0.77 and passed the 0.75 gate silently."""
        assert normalise_name("Priya Menon I need a cleaning") == "Priya Menon"
        assert normalise_name("Sarah Thomas and I need a root canal") == "Sarah Thomas"

    def test_indic_script_names_are_not_case_mangled(self):
        assert normalise_name("അഞ്ജലി") == "അഞ്ജലി"


class TestSlotConfidence:
    def test_phone_threshold_is_the_strictest(self):
        """One wrong digit stays a valid number - nothing downstream can catch it."""
        assert CONFIRMATION_THRESHOLDS["phone"] > CONFIRMATION_THRESHOLDS["patient_name"]
        assert CONFIRMATION_THRESHOLDS["phone"] > CONFIRMATION_THRESHOLDS["procedure"]

    def test_low_asr_confidence_forces_readback(self):
        s = extract_slots("my name is Priya Menon", REF, {"priya": 0.4, "menon": 0.4})
        assert s.patient_name.needs_confirmation

    def test_high_asr_confidence_passes(self):
        s = extract_slots("I need a cleaning", REF, {"cleaning": 0.99})
        assert not s.procedure.needs_confirmation

    def test_vague_time_is_discounted(self):
        vague = extract_slots("tomorrow morning", REF).appointment_time
        exact = extract_slots("tomorrow at 3 pm", REF).appointment_time
        assert vague.confidence < exact.confidence

    def test_not_bookable_until_every_slot_usable(self):
        s = extract_slots(
            "my name is Priya Menon I need a cleaning tomorrow at 3 pm my number is "
            "nine seven one five zero one two three four five six seven", REF,
            {"priya": 0.4, "menon": 0.4},
        )
        assert not s.bookable
        s.patient_name.confirm()
        for slot in s.pending_confirmation:
            slot.confirm()
        assert s.bookable

    def test_rejection_clears_rather_than_keeping_a_guess(self):
        s = extract_slots("my name is Priya Menon", REF, {"priya": 0.4, "menon": 0.4})
        s.patient_name.reject()
        assert s.patient_name.value is None
        assert not s.patient_name.filled

    def test_confirmed_slot_is_not_overwritten(self):
        s = extract_slots("my name is Priya Menon", REF)
        s.patient_name.confirm()
        s = extract_slots("my name is Someone Else", REF, existing=s)
        assert s.patient_name.value == "Priya Menon"

    def test_readback_prompts_are_slot_specific(self):
        s = extract_slots("my number is nine seven one five zero one two three four five six seven", REF)
        assert "confirm your number" in readback_prompt(s.phone)


def test_name_capture_stops_at_the_procedure():
    """Regression: "my name is Sara Ali root canal on 15/08" produced the
    patient name "Sara Ali Root Canal" at 0.765 - above the 0.75 gate, so it
    booked and was sent on the WhatsApp confirmation with no read-back.

    Every corpus utterance happens to put a stop-word between the name and the
    procedure ("I need a cleaning"), so a caller who does not pause that way
    was never covered.
    """
    from datetime import datetime as _dt
    from receptionist.nlu.slots import extract_slots as _extract

    slots = _extract("my name is Sara Ali root canal on 15/08 at 11am number 0509998887",
                     _dt(2026, 7, 27, 10, 0))
    assert slots.patient_name.value == "Sara Ali"
    assert slots.procedure.value == "root_canal"


def test_name_boundary_covers_every_procedure_word():
    """The two lists cannot drift: a procedure keyword that is not a name
    boundary is a name that swallows it."""
    from receptionist.nlu.normalize import NAME_BOUNDARY
    from receptionist.nlu.slots import PROCEDURES

    for keywords in PROCEDURES.values():
        for keyword in keywords:
            for word in keyword.split():
                assert word in NAME_BOUNDARY, (
                    f"procedure word {word!r} would be absorbed into a patient name"
                )
