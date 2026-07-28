"""LLM-primary extraction: what the model decides, and what it must not.

No test here reaches a model. The chat model is injected, so the suite stays
offline - which is also the property that keeps the clinic answering the
phone when Ollama is down.
"""
from datetime import datetime

import pytest

from receptionist.nlu.llm_slots import (
    CallSlots,
    build_model,
    is_cloud_model,
    extract_slots_llm,
    is_local,
)
from receptionist.nlu.slots import SlotSet

NOW = datetime(2026, 7, 28, 10, 0)          # a Tuesday


class FakeModel:
    def __init__(self, result=None, raises=None):
        self._result, self._raises = result, raises
        self.prompts = []

    def with_structured_output(self, schema, **kwargs):
        self.method = kwargs.get("method")
        return self

    def invoke(self, messages):
        self.prompts.append(messages)
        if self._raises:
            raise self._raises
        return self._result


def run(text, result, **kw):
    return extract_slots_llm(text, NOW, FakeModel(result), **kw)


# ------------------------------------------------- privacy is structural
def test_a_remote_host_is_refused_unless_chosen():
    """The transcript is a patient's name, number and complaint in one
    sentence. Sending it off the machine is allowed, but has to be a
    decision rather than a default."""
    with pytest.raises(ValueError, match="leave this machine"):
        build_model("qwen3:4b", "https://example.com:11434")


def test_a_cloud_model_is_remote_even_though_it_is_served_via_localhost():
    """The hole a URL check misses. Ollama Cloud models are proxied by the
    local daemon, so the base URL is localhost and the inference is not - a
    guard reading only the URL would have passed every patient's name to
    ollama.com while reporting that nothing left the machine."""
    assert is_cloud_model("gpt-oss:120b-cloud")
    assert not is_cloud_model("qwen3:4b")
    with pytest.raises(ValueError, match="cloud model"):
        build_model("gpt-oss:120b-cloud", "http://localhost:11434")


def test_remote_is_permitted_when_explicitly_allowed():
    build_model("gpt-oss:120b-cloud", "http://localhost:11434", allow_remote=True)


def test_localhost_variants_are_recognised():
    for url in ("http://localhost:11434", "http://127.0.0.1:11434",
                "http://host.docker.internal:11434"):
        assert is_local(url)
    for url in ("https://api.openai.com", "https://ollama.com", ""):
        assert not is_local(url)


# ------------------------------------------------- the rules' failures, fixed
def test_a_bare_name_is_understood():
    """No "my name is" trigger. The rule extractor matched nothing and asked
    three more times."""
    slots = run("Amna Ansari", CallSlots(patient_name="Amna Ansari"),
                awaiting="patient_name")
    assert slots.patient_name.value == "Amna Ansari"


def test_a_symptom_sentence_is_not_read_as_a_name():
    """"I am having ache in my left jaw" produced the patient name "Having"."""
    slots = run("I am having ache in my left jaw",
                CallSlots(patient_name=None, procedure="checkup"))
    assert slots.patient_name.value is None
    assert slots.procedure.value == "checkup"


def test_a_described_symptom_becomes_a_checkup_and_is_read_back():
    """A phone description cannot distinguish a wisdom tooth that needs
    removing from one that needs an X-ray."""
    slots = run("my wisdom tooth is aching", CallSlots(procedure="checkup"))
    assert slots.procedure.value == "checkup"
    assert slots.procedure.source == "symptom"
    assert slots.procedure.needs_confirmation


def test_a_named_procedure_is_scored_higher_than_an_inferred_one():
    named = run("I need a cleaning", CallSlots(procedure="cleaning"))
    inferred = run("my tooth is sore", CallSlots(procedure="checkup"))
    assert named.procedure.confidence > inferred.procedure.confidence


# ------------------------------------------------- confidence is not the model's
def test_confidence_comes_from_the_audio_not_the_model():
    """The model says what was said; the acoustics say how sure we are it was
    heard. A decoder's own probability is highest exactly when it is fluently
    wrong."""
    clear = run("my name is Priya Menon", CallSlots(patient_name="Priya Menon"),
                word_confidences={"priya": 0.97, "menon": 0.97})
    muffled = run("my name is Priya Menon", CallSlots(patient_name="Priya Menon"),
                  word_confidences={"priya": 0.30, "menon": 0.30})
    assert clear.patient_name.confidence > muffled.patient_name.confidence
    assert muffled.patient_name.needs_confirmation


def test_badly_heard_digits_never_book():
    slots = run("my number is 0501234567", CallSlots(phone="0501234567"),
                word_confidences={"0501234567": 0.20})
    assert slots.phone.value == "+971501234567"
    assert not slots.phone.usable


def test_a_past_time_is_penalised():
    slots = run("yesterday at 3 pm",
                CallSlots(appointment_datetime="2026-07-27T15:00"))
    assert slots.appointment_time.needs_confirmation
    assert "past" in " ".join(slots.appointment_time.notes)


# ------------------------------------------- the model does not do arithmetic
def test_the_calendar_overrules_the_model_on_which_day_saturday_is():
    """Measured, not hypothetical: the model answered "Saturday at 2 pm" with
    2026-07-31 - a Friday - at confidence 0.942, because the acoustics were
    clean and acoustic confidence is what scores this slot. Nothing downstream
    could catch it: the caller was heard perfectly and would have been booked
    into the wrong day without being asked.

    So the day is counted by `normalise_datetime`, which is deterministic, and
    the model is only asked which words the caller used.
    """
    slots = run("Saturday at 2 pm",
                CallSlots(appointment_datetime="2026-07-31T14:00",
                          spoken_time_phrase="Saturday at 2 pm"),
                word_confidences={"saturday": 0.99, "2": 0.99, "pm": 0.99})
    assert slots.appointment_time.value == datetime(2026, 8, 1, 14, 0)


def test_a_disagreement_neither_side_wins_is_read_back_rather_than_booked():
    """No weekday was named, so nothing arbitrates. Invariant 5: the
    disagreement is itself the evidence that one of them is wrong."""
    slots = run("tomorrow at 2 pm",
                CallSlots(appointment_datetime="2026-07-30T14:00",
                          spoken_time_phrase="tomorrow at 2 pm"),
                word_confidences={"tomorrow": 0.99, "2": 0.99, "pm": 0.99})
    assert slots.appointment_time.needs_confirmation
    assert not slots.bookable


def test_agreement_is_left_alone():
    """The check must not tax the common case, or it becomes a caller asked to
    confirm every appointment they ever booked correctly."""
    slots = run("Saturday at 2 pm",
                CallSlots(appointment_datetime="2026-08-01T14:00",
                          spoken_time_phrase="Saturday at 2 pm"),
                word_confidences={"saturday": 0.99, "2": 0.99, "pm": 0.99})
    assert slots.appointment_time.value == datetime(2026, 8, 1, 14, 0)
    assert not slots.appointment_time.needs_confirmation


def test_the_model_wins_when_the_rule_is_the_one_that_is_wrong():
    """The split cuts both ways, which is why neither side is simply trusted.
    `normalise_datetime` does not decline a phrase it half understands - it
    reads "the day after next Thursday" as plain Thursday and is a week out.
    Both land on a Thursday, so the weekday cannot arbitrate and the model's
    reading is kept - but the caller is still asked."""
    slots = run("the day after next Thursday at 4",
                CallSlots(appointment_datetime="2026-08-06T16:00",
                          spoken_time_phrase="the day after next Thursday at 4"))
    assert slots.appointment_time.value == datetime(2026, 8, 6, 16, 0)
    assert slots.appointment_time.needs_confirmation


def test_a_phrase_with_no_hour_is_still_not_an_appointment():
    """"Saturday morning" resolves to a day. Inventing an hour for it is the
    day-of-month-as-hour bug in a new costume."""
    slots = run("Saturday morning",
                CallSlots(appointment_datetime=None,
                          spoken_time_phrase="Saturday morning"))
    assert slots.appointment_time.value is None


# ------------------------------------------------- refusals and fallback
def test_a_vague_time_is_left_unset():
    """"Saturday morning" is a day, not an appointment. The model is told to
    return null and the value is not invented downstream either."""
    slots = run("Saturday morning", CallSlots(appointment_datetime=None))
    assert slots.appointment_time.value is None


def test_an_unparseable_phone_is_dropped_rather_than_stored():
    slots = run("my number is banana", CallSlots(phone="banana"))
    assert slots.phone.value is None


def test_an_unreachable_model_returns_none_so_the_rules_take_over():
    """A clinic must still answer the phone when Ollama is down."""
    for failure in (TimeoutError("timed out"), ConnectionError("refused"),
                    ValueError("bad structured output")):
        assert extract_slots_llm("hello", NOW, FakeModel(raises=failure)) is None


def test_a_confirmed_slot_is_never_overwritten():
    existing = SlotSet()
    slots = run("my name is Priya Menon", CallSlots(patient_name="Priya Menon"),
                existing=existing)
    slots.patient_name.confirm()
    again = run("this is Someone Else", CallSlots(patient_name="Someone Else"),
                existing=slots)
    assert again.patient_name.value == "Priya Menon"


def test_the_model_is_told_what_the_agent_just_asked_for():
    """Context is the cheapest signal available and the rules ignored it."""
    model = FakeModel(CallSlots(patient_name="Amna Ansari"))
    extract_slots_llm("Amna Ansari", NOW, model, awaiting="patient_name")
    assert "patient_name" in str(model.prompts[0])


def test_structured_output_uses_tool_calling():
    """json_schema mode was ignored outright by the hosted model, which
    answered under invented field names."""
    model = FakeModel(CallSlots())
    extract_slots_llm("hello", NOW, model)
    assert model.method == "function_calling"
