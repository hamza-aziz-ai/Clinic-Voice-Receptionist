"""LLM-primary slot extraction.

WHY THIS REPLACED THE RULES AS THE UNDERSTANDING LAYER

The rule extractor is a hand-written slot filler, and it failed the way those
always fail. Real calls, all in one afternoon:

  * "I am having ache in my left jaw" became the patient name "Having"
  * "my wisdom tooth is not coming up properly" matched no procedure at all
  * "Amna Ansari", answering "could I take your full name?", matched nothing
    because there was no "my name is" in front of it

Each was fixed with another regex, another stop-word list, another special
case. That is the treadmill this file exists to get off. A language model
handles all three without being told, because understanding what a sentence
means is the thing it is actually good at.

WHAT DID *NOT* MOVE TO THE MODEL

Confidence. The model says **what was said**; the acoustics say **how sure we
are it was heard**. Those are different questions with different evidence,
and conflating them is the mistake the whole project is built to avoid: a
sequence-to-sequence decoder's own probability is a statement about its
output, not about the audio, and it is highest exactly when the model is
fluently wrong.

So every value the model returns is scored from the ASR word confidences over
the span it came from - the same function the rules use - and then passed
through the same read-back gate. The model can be as clever as it likes about
meaning and still cannot book something the microphone did not clearly hear.

WHY NO REDACTION HERE

The cross-check sends transcripts to a *hosted* model, so names and numbers
are replaced with surrogates first - which is also why it could only ever
compare two of the four slots. This path requires a **local** model: nothing
leaves the machine, so the model sees the real utterance and can extract all
four. ``require_local`` enforces that rather than trusting configuration.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from .normalize import normalise_datetime, normalise_name, normalise_phone
from .slots import (
    CONFIRMATION_THRESHOLDS,
    PROCEDURES,
    Slot,
    SlotSet,
    _asr_confidence,
)

log = logging.getLogger(__name__)

NOT_STATED = "none"

# Anything not on localhost is a third party, and a transcript is a patient's
# name, number and medical complaint in one sentence.
LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1", "host.docker.internal")

# A value the model read but the rules did not confirm is believed and read
# back rather than believed and booked. Understanding is not hearing.
UNCORROBORATED_FACTOR = 0.88


class CallSlots(BaseModel):
    """What the model is asked for. Nullable throughout, on purpose.

    A model forced to produce a value invents one, and an invented phone
    number is the single worst thing this system can produce.
    """

    patient_name: str | None = Field(
        None, description="Full name of the caller, exactly as spoken. Null if not said."
    )
    phone: str | None = Field(
        None, description="Phone number, digits as spoken. Null if not said."
    )
    appointment_datetime: str | None = Field(
        None,
        description=(
            "Requested date and time as ISO 8601 (YYYY-MM-DDTHH:MM), resolved "
            "against the reference time. Null if no time of day was stated - "
            "'Saturday morning' is a day, not a time, so that is null."
        ),
    )
    procedure: Literal[
        "cleaning", "extraction", "root_canal", "filling",
        "checkup", "whitening", "braces", NOT_STATED,
    ] | None = Field(None, description="Treatment requested, or none.")


SYSTEM_PROMPT = """You extract appointment details from dental clinic calls.

The caller may speak English, Tamil, Kannada, Malayalam or Hindi, and often \
mixes an Indic language with English clinical terms in one sentence.

Extract only what the caller actually said. Never invent a value; null is \
always allowed and is the right answer when something was not stated.

Names: take the caller's name however it arrives. "My name is Amna Ansari", \
"Amna Ansari", and "this is Amna" all give a name. A reply that is a \
greeting, a confirmation, a day of the week or a symptom is NOT a name.

Times: resolve relative dates against the reference time. If no time of day \
was spoken, appointment_datetime is null - "Saturday morning" and "tomorrow" \
are days, not appointments. Never pick an hour the caller did not say.

Procedure: map what they describe onto one code. Callers describe symptoms \
rather than treatments, and a described symptom is always "checkup" - you \
cannot tell from a phone call whether an aching wisdom tooth needs removing \
or an X-ray, and booking an extraction on a guess is a clinical decision \
nobody authorised.
  cleaning   - cleaning, scaling, polish, hygiene
  extraction - the caller explicitly asks for a tooth to be removed
  root_canal - root canal, RCT, endo
  filling    - filling, cavity
  checkup    - checkup, consultation, and ALL described symptoms: pain,
               aching, swelling, sensitivity, a wisdom tooth, a chipped tooth
  whitening  - whitening, bleaching
  braces     - braces, aligners, orthodontic
"""

USER_PROMPT = """Reference time (the moment of the call): {reference_time}
The agent has just asked the caller for: {awaiting}

Caller said:
{transcript}
"""


def is_local(base_url: str) -> bool:
    lowered = (base_url or "").lower()
    return any(host in lowered for host in LOCAL_HOSTS)


def build_local_model(model: str, base_url: str, timeout_s: float = 20.0) -> Any:
    """Chat model for extraction. Refuses a remote host.

    Enforced here rather than documented, because the failure is silent: a
    hosted endpoint would work perfectly and quietly ship patient names to a
    third party on every turn.
    """
    if not is_local(base_url):
        raise ValueError(
            f"LLM extraction requires a local model; {base_url!r} is remote and "
            "the transcript contains patient names and phone numbers"
        )
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0.0,
        # Reasoning models emit a long chain of thought before answering.
        # qwen3:4b took over 60 seconds to extract four fields from one
        # sentence, which is not a conversation. There is nothing to reason
        # about here - the utterance is one line and the schema is four
        # fields.
        reasoning=False,
        # THE SINGLE BIGGEST LATENCY FACTOR HERE.
        #
        # Ollama defaults qwen3 to a 131k context window, which inflates a
        # 2.5 GB model to 23 GB of allocation - far past a 6 GB laptop GPU, so
        # 84% of it runs on the CPU and one extraction takes over two minutes.
        # `ollama ps` shows it plainly: "23 GB  84%/16% CPU/GPU".
        #
        # The prompt here is a system message and one utterance. 2048 tokens
        # is generous, keeps the whole model resident in VRAM, and is the
        # difference between a conversation and a timeout.
        num_ctx=2048,
        # A four-field object is small. Without a cap a model that starts
        # rambling holds the whole turn hostage.
        num_predict=256,
        # Keep it resident between turns; reloading 2.5 GB per utterance
        # would dominate the latency budget.
        keep_alive="30m",
        client_kwargs={"timeout": timeout_s},
    )


def extract_slots_llm(
    transcript: str,
    reference_time: datetime,
    chat_model: Any,
    word_confidences: dict[str, float] | None = None,
    existing: SlotSet | None = None,
    awaiting: str | None = None,
) -> SlotSet | None:
    """Understand the utterance with the model, score it from the audio.

    Returns None when the model is unreachable or answers unusably, so the
    caller can fall back to the rules. A clinic must still answer the phone
    when Ollama is down.
    """
    slots = existing or SlotSet()
    try:
        structured = chat_model.with_structured_output(
            CallSlots, method="function_calling"
        )
        result = structured.invoke([
            ("system", SYSTEM_PROMPT),
            ("user", USER_PROMPT.format(
                reference_time=reference_time.strftime("%Y-%m-%d %H:%M (%A)"),
                awaiting=awaiting or "nothing in particular",
                transcript=transcript,
            )),
        ])
    except Exception as exc:
        log.warning("LLM extraction unavailable: %s: %s", type(exc).__name__, exc)
        return None
    if result is None:
        return None

    _apply_name(slots, result, transcript, word_confidences)
    _apply_phone(slots, result, transcript, word_confidences)
    _apply_time(slots, result, transcript, reference_time, word_confidences)
    _apply_procedure(slots, result, transcript, word_confidences)
    return slots


def _score(span: str, transcript: str, word_confidences: dict[str, float] | None) -> float:
    """Confidence from the acoustics, never from the model.

    Scored over the span the value came from when those words appear in the
    transcript, falling back to the whole utterance. The model's certainty
    about its own output is deliberately not consulted.
    """
    probe = span if span and span.lower() in transcript.lower() else transcript
    return _asr_confidence(word_confidences, probe)


def _keep(slot: Slot, candidate: Slot) -> bool:
    """Same rule the rule extractor uses: never overwrite a trusted value."""
    if slot.confirmed:
        return False
    if not slot.filled:
        return True
    if not slot.needs_confirmation:
        return False
    return candidate.confidence > slot.confidence


def _apply_name(slots, result, transcript, word_confidences) -> None:
    raw = (result.patient_name or "").strip()
    if not raw:
        return
    # Still normalised: the model returns what was said, and the sanitiser is
    # what makes it safe to print and store.
    parsed = normalise_name(raw) or raw
    conf = _score(raw, transcript, word_confidences) * 0.90
    if raw.lower() not in transcript.lower():
        # Not literally present - the model inferred or reformatted it.
        conf *= UNCORROBORATED_FACTOR
    candidate = Slot("patient_name", raw_text=raw, value=parsed,
                     confidence=max(0.0, min(1.0, conf)), source="llm")
    if _keep(slots.patient_name, candidate):
        slots.patient_name = candidate


def _apply_phone(slots, result, transcript, word_confidences) -> None:
    raw = (result.phone or "").strip()
    if not raw:
        return
    parsed = normalise_phone(raw)
    if not parsed:
        return
    conf = (_score(raw, transcript, word_confidences) - parsed.confidence_penalty) * 0.95
    candidate = Slot("phone", raw_text=raw, value=parsed.e164,
                     confidence=max(0.0, min(1.0, conf)), source="llm",
                     notes=[f"country {parsed.country}"])
    if _keep(slots.phone, candidate):
        slots.phone = candidate


def _apply_time(slots, result, transcript, reference_time, word_confidences) -> None:
    raw = (result.appointment_datetime or "").strip()
    if not raw:
        return
    try:
        when = datetime.fromisoformat(raw.replace("Z", ""))
    except ValueError:
        parsed = normalise_datetime(raw, reference_time)
        if not parsed or parsed.time_was_vague:
            return
        when = parsed.when
    conf = _score(raw, transcript, word_confidences)
    notes = []
    if when < reference_time:
        conf *= 0.5
        notes.append("resolved to a past time")
    candidate = Slot("appointment_time", raw_text=raw, value=when,
                     confidence=max(0.0, min(1.0, conf)), source="llm", notes=notes)
    if _keep(slots.appointment_time, candidate):
        slots.appointment_time = candidate


def _apply_procedure(slots, result, transcript, word_confidences) -> None:
    code = (result.procedure or "").strip().lower()
    if not code or code == NOT_STATED or code not in PROCEDURES:
        return
    # Score on the keyword if the caller used one; otherwise the model
    # inferred it from a description, which is weaker evidence and is read
    # back. "Aching" is not the caller asking for a check-up.
    hit = next((k for k in PROCEDURES[code] if k in transcript.lower()), None)
    conf = _score(hit or transcript, transcript, word_confidences)
    notes = [f"matched {hit!r}"] if hit else ["inferred from a described symptom"]
    if not hit:
        conf *= 0.75
    candidate = Slot("procedure", raw_text=hit or code, value=code,
                     confidence=max(0.0, min(1.0, conf)),
                     source="llm" if hit else "symptom", notes=notes)
    if _keep(slots.procedure, candidate):
        slots.procedure = candidate
