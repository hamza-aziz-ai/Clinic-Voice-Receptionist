"""Redaction, the LLM extractor's failure behaviour, and the cross-check.

No test here reaches the network. The chat model is injected, so the whole
file runs on a machine with no Ollama - which is also the property that keeps
booking working when Ollama is down.
"""
from datetime import datetime

import pytest

from receptionist.nlu.crosscheck import (
    COMPARABLE_SLOTS,
    DISAGREEMENT_CEILING,
    apply_crosscheck,
)
from receptionist.nlu.llm_extractor import (
    ExtractedSlots,
    LLMExtraction,
    extract_llm_slots,
)
from receptionist.nlu.redaction import (
    SURROGATE_NAMES,
    SURROGATE_PHONES,
    redact,
    residual_identifiers,
    script_of,
)
from receptionist.nlu.slots import CONFIRMATION_THRESHOLDS, extract_slots

NOW = datetime(2026, 7, 27, 10, 0)
UTTERANCE = ("my name is Priya Menon I need a cleaning tomorrow at 3 pm "
             "my number is 0501234567")


class FakeModel:
    """Stands in for ChatOllama. Records what it was asked."""

    def __init__(self, result=None, raises=None):
        self._result = result
        self._raises = raises
        self.prompts = []

    def with_structured_output(self, schema, **kwargs):
        self.method = kwargs.get("method")
        return self

    def invoke(self, messages):
        self.prompts.append(messages)
        if self._raises:
            raise self._raises
        return self._result


def rule_slots(text=UTTERANCE, now=NOW):
    return extract_slots(text, now)


# ------------------------------------------------------------- redaction
def test_real_identifiers_never_appear_in_the_outgoing_text():
    slots = rule_slots()
    r = redact(UTTERANCE, slots.patient_name.value, slots.phone.raw_text)
    assert "Priya" not in r.text
    assert "Menon" not in r.text
    assert "0501234567" not in r.text


def test_surrogates_map_back_exactly():
    slots = rule_slots()
    r = redact(UTTERANCE, slots.patient_name.value, slots.phone.raw_text)
    for surrogate, original in r.reverse.items():
        assert r.restore(surrogate) == original


def test_restore_leaves_unknown_values_alone():
    r = redact(UTTERANCE, "Priya Menon", "0501234567")
    assert r.restore("something else") == "something else"
    assert r.restore(None) is None


def test_surrogate_choice_is_stable_across_processes():
    """Built-in hash() is salted per process; the same name must not get a
    different surrogate on every run."""
    a = redact(UTTERANCE, "Priya Menon", None)
    b = redact(UTTERANCE, "Priya Menon", None)
    assert a.text == b.text


def test_indic_name_gets_an_indic_surrogate():
    """A Latin surrogate in a Malayalam utterance changes the segmentation
    problem and the language evidence - the opposite of a faithful stand-in."""
    assert script_of("അഞ്ജലി നായർ") == "ml"
    r = redact("എന്റെ പേര് അഞ്ജലി നായർ", "അഞ്ജലി നായർ", None)
    surrogate = next(iter(r.reverse))
    assert surrogate in SURROGATE_NAMES["ml"]


def test_script_detection_survives_combining_marks():
    """Mn/Mc characters are not isalpha(); the range check must not care."""
    assert script_of("കുമാർ") == "ml"
    assert script_of("கார்த்திக்") == "ta"
    assert script_of("ಸುರೇಶ್") == "kn"
    assert script_of("शर्मा") == "hi"


def test_residual_scan_ignores_our_own_surrogates():
    assert residual_identifiers("call +971501110001", {"+971501110001"}) == []


def test_residual_scan_catches_a_number_the_rules_missed():
    leftovers = residual_identifiers("also try 0559998888", {"+971501110001"})
    assert leftovers


def test_a_second_number_blocks_the_send_entirely():
    """A digit run the rule extractor did not find is a real mobile that
    would land on a third party's server. Refuse rather than send."""
    text = ("my name is Priya Menon call me on 0501234567 "
            "or my husband on 0559998888")
    slots = extract_slots(text, NOW)
    model = FakeModel(result=ExtractedSlots())
    out = extract_llm_slots(text, NOW, slots.patient_name.value,
                            slots.phone.raw_text, chat_model=model)
    assert out is None
    assert model.prompts == [], "nothing may be sent when a leak is detected"


def test_an_identifier_that_cannot_be_found_blocks_the_send():
    """The failure that has no downstream catcher: nothing scans for names,
    so a redaction that silently no-ops posts a real patient to a third party.
    """
    r = redact("my name is Priya Menon", name="Priya  Menon")   # double space
    assert r.unremoved == ["Priya  Menon"]

    model = FakeModel(result=ExtractedSlots())
    out = extract_llm_slots("my name is Priya Menon", NOW,
                            name="Priya  Menon", chat_model=model)
    assert out is None
    assert model.prompts == []


def test_the_name_span_is_not_what_gets_redacted():
    """NAME_SPAN captures up to forty characters and is trimmed by
    normalisation, so redacting on raw_text ate the procedure and the time."""
    slots = rule_slots()
    assert slots.patient_name.raw_text != slots.patient_name.value
    assert "cleaning" in slots.patient_name.raw_text

    r = redact(UTTERANCE, slots.patient_name.value, slots.phone.raw_text)
    assert "cleaning" in r.text and "3 pm" in r.text


def test_the_prompt_actually_sent_contains_no_real_identifier():
    slots = rule_slots()
    model = FakeModel(result=ExtractedSlots(procedure="cleaning"))
    extract_llm_slots(UTTERANCE, NOW, slots.patient_name.value,
                      slots.phone.raw_text, chat_model=model)
    sent = str(model.prompts[0])
    assert "Priya" not in sent and "0501234567" not in sent


# ------------------------------------------------------------- failure modes
@pytest.mark.parametrize("failure", [
    TimeoutError("timed out"),
    ConnectionRefusedError("ollama not running"),
    ValueError("could not parse structured output"),
    RuntimeError("model not found"),
])
def test_every_llm_failure_returns_none_and_never_raises(failure):
    """A receptionist that stops booking because a language model is
    unreachable is worse than one that never had a language model."""
    slots = rule_slots()
    out = extract_llm_slots(UTTERANCE, NOW, slots.patient_name.value,
                            slots.phone.raw_text,
                            chat_model=FakeModel(raises=failure))
    assert out is None


def test_structured_output_uses_tool_calling_not_json_schema():
    """With LangChain's default json_schema method, gpt-oss:120b-cloud ignored
    the schema and answered under invented field names - caller_name,
    treatment_code - failing validation on every call while having the right
    answer. Ollama's `format` is not enforced for this cloud-proxied model."""
    from receptionist.nlu.llm_extractor import STRUCTURED_OUTPUT_METHOD
    assert STRUCTURED_OUTPUT_METHOD == "function_calling"

    slots = rule_slots()
    model = FakeModel(result=ExtractedSlots(procedure="cleaning"))
    extract_llm_slots(UTTERANCE, NOW, slots.patient_name.value,
                      slots.phone.raw_text, chat_model=model)
    assert model.method == "function_calling"


def test_absent_values_are_accepted_as_null_or_sentinel():
    """Tool-calling emits null; a flat-enum route emits "none". Both mean the
    caller did not say it, and both must arrive downstream as None."""
    slots = rule_slots()
    for raw in (None, "none"):
        model = FakeModel(result=ExtractedSlots(procedure=raw, patient_name=None))
        out = extract_llm_slots(UTTERANCE, NOW, slots.patient_name.value,
                                slots.phone.raw_text, chat_model=model)
        assert out.procedure is None
        assert out.patient_name is None


def test_the_schema_itself_rejects_an_off_schema_procedure():
    """Asked for a free string with the options in the description, the model
    returned "procedure_cleaning" on every live call - it invented a naming
    convention from the field name. A Literal becomes a JSON-schema enum,
    which constrains generation instead of requesting politely."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ExtractedSlots(procedure="dental implant")
    with pytest.raises(ValidationError):
        ExtractedSlots(procedure="procedure_cleaning")


def test_a_validation_failure_costs_the_crosscheck_not_the_booking():
    """with_structured_output raises when the model breaks the schema. That
    must degrade to no second opinion, never to a failed call."""
    from pydantic import ValidationError
    slots = rule_slots()
    failure = ValidationError.from_exception_data("ExtractedSlots", [])
    out = extract_llm_slots(UTTERANCE, NOW, slots.patient_name.value,
                            slots.phone.raw_text,
                            chat_model=FakeModel(raises=failure))
    assert out is None


def test_model_answers_are_mapped_back_to_real_values():
    slots = rule_slots()
    redaction = redact(UTTERANCE, slots.patient_name.value, slots.phone.raw_text)
    surrogate_name = next(s for s, o in redaction.reverse.items() if o == "Priya Menon")
    model = FakeModel(result=ExtractedSlots(patient_name=surrogate_name))
    out = extract_llm_slots(UTTERANCE, NOW, slots.patient_name.value,
                            slots.phone.raw_text, chat_model=model)
    assert out.patient_name == "Priya Menon"


# ------------------------------------------------------------- cross-check
def test_no_extraction_leaves_every_confidence_untouched():
    slots = rule_slots()
    before = [s.confidence for s in slots.all_slots()]
    report = apply_crosscheck(slots, None)
    assert not report.available
    assert [s.confidence for s in slots.all_slots()] == before


def test_agreement_never_raises_confidence():
    """Both extractors read the same degraded transcript, so their errors are
    correlated. Agreement is not evidence."""
    slots = rule_slots()
    slots.procedure.confidence = 0.72
    apply_crosscheck(slots, LLMExtraction(procedure="cleaning"))
    assert slots.procedure.confidence == 0.72


def test_disagreement_forces_a_readback():
    slots = rule_slots()
    slots.procedure.confidence = 0.99
    report = apply_crosscheck(slots, LLMExtraction(procedure="filling"))
    assert "procedure" in report.disagreed
    assert slots.procedure.confidence == DISAGREEMENT_CEILING
    assert slots.procedure.needs_confirmation


def test_the_ceiling_sits_below_every_threshold():
    """A disagreement must always trigger the read-back, never just nudge."""
    assert all(DISAGREEMENT_CEILING < t for t in CONFIRMATION_THRESHOLDS.values())


def test_datetime_disagreement_is_caught():
    slots = rule_slots()
    slots.appointment_time.confidence = 0.99
    apply_crosscheck(slots, LLMExtraction(appointment_datetime="2026-07-28T16:00"))
    assert slots.appointment_time.confidence == DISAGREEMENT_CEILING


def test_a_minute_of_drift_is_not_a_disagreement():
    slots = rule_slots()
    slots.appointment_time.confidence = 0.99
    apply_crosscheck(slots, LLMExtraction(appointment_datetime="2026-07-28T15:00:30"))
    assert slots.appointment_time.confidence == 0.99


def test_unparseable_datetime_is_not_treated_as_disagreement():
    """The model failed to answer in the format asked for. That is our prompt's
    problem, not evidence about the appointment."""
    slots = rule_slots()
    slots.appointment_time.confidence = 0.99
    report = apply_crosscheck(slots, LLMExtraction(appointment_datetime="tomorrow 3pm"))
    assert slots.appointment_time.confidence == 0.99
    assert "appointment_time" in report.skipped


def test_a_confirmed_slot_outranks_the_model():
    slots = rule_slots()
    slots.procedure.confirm()
    slots.procedure.confidence = 0.99
    apply_crosscheck(slots, LLMExtraction(procedure="braces"))
    assert slots.procedure.confidence == 0.99


def test_redacted_fields_are_never_compared():
    """Agreement on a surrogate we injected says the model can copy, not that
    the digits are right. Comparing them would manufacture reassurance."""
    assert "patient_name" not in COMPARABLE_SLOTS
    assert "phone" not in COMPARABLE_SLOTS

    slots = rule_slots()
    slots.phone.confidence = 0.99
    slots.patient_name.confidence = 0.99
    apply_crosscheck(slots, LLMExtraction(patient_name="Someone Else",
                                          phone="+971559999999"))
    assert slots.phone.confidence == 0.99
    assert slots.patient_name.confidence == 0.99


def test_surrogate_pools_hold_no_real_patient_data():
    """Guards against someone pasting a live number in as a 'realistic' sample."""
    for pool in SURROGATE_NAMES.values():
        assert len(set(pool)) == len(pool)
    assert all(p.startswith("+971") or p.startswith("+91") for p in SURROGATE_PHONES)
    assert len(set(SURROGATE_PHONES)) == len(SURROGATE_PHONES)
