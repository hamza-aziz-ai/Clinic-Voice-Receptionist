"""Second-opinion slot extraction via LangChain and Ollama.

The rule extractor stays authoritative. This one exists to *disagree* - to
notice when the regexes were confident and wrong, which is the only error
class that reaches a patient.

WHAT THIS DELIBERATELY DOES NOT DO

It does not report its own confidence. A model asked "how sure are you?"
returns a number produced by the same process that produced the answer, and
a wrong answer arrives with a wrong confidence attached. The confidence in
this system is computed from ASR spans, normalisation outcomes and now
inter-extractor agreement - all of which are measurable from outside the
thing being judged.

It does not get a vote on the value either. Its answer never overwrites a
slot. See ``crosscheck.py``: disagreement lowers confidence, which triggers
a read-back, which asks the human who actually knows.

FAILURE IS ROUTINE, NOT EXCEPTIONAL

Ollama is a network call to a remote host. It times out, it returns prose
where JSON was asked for, the model gets pulled. Every one of those has to
leave the system exactly as it was without the cross-check - a receptionist
that stops booking because a language model is unreachable is worse than one
that never had a language model. ``extract_llm_slots`` returns None on any
failure and never raises.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from .redaction import redact, residual_identifiers

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-oss:120b-cloud"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT_S = 30.0

PROCEDURE_CODES = (
    "cleaning", "extraction", "root_canal", "filling",
    "checkup", "whitening", "braces",
)


# Sentinel for "the caller did not say this". A model forced to produce a
# real value invents one, so "not stated" has to stay expressible - but it
# cannot be expressed as null. See ExtractedSlots.
NOT_STATED = "none"


# How the schema is imposed. Not the default, and the difference is not
# cosmetic: with LangChain's default "json_schema" method, gpt-oss:120b-cloud
# ignored the schema completely and answered under field names of its own
# invention - {"caller_name": ..., "treatment_code": "cleaning"} - which
# failed validation on every single call. Ollama's `format` parameter is not
# enforced for this cloud-proxied model, so the schema was arriving as a
# suggestion. Tool-calling puts the field names in the function signature,
# and the same prompt then came back correctly shaped.
#
# Worth noting what the failure looked like from outside: the model had the
# right answer ("cleaning") and the integration threw it away. A cross-check
# that silently returns nothing looks identical to a cross-check that agrees.
STRUCTURED_OUTPUT_METHOD = "function_calling"


class ExtractedSlots(BaseModel):
    """Schema the model is constrained to.

    Fields are nullable *and* carry a "not stated" sentinel, because the two
    routes disagree about how to say "absent": tool-calling emits null, and a
    model told to use a flat enum emits the sentinel. Accepting both costs one
    line at the boundary and removes a class of validation failure that
    presents as silent disagreement.
    """

    patient_name: str | None = Field(
        None, description="Full name exactly as it appears, or null if not stated."
    )
    phone: str | None = Field(
        None, description="Phone number exactly as it appears, or null if not stated."
    )
    appointment_datetime: str | None = Field(
        None,
        description=(
            "Requested date and time as ISO 8601 (YYYY-MM-DDTHH:MM), resolved "
            "against the reference time given. Null if no time was requested, "
            "or if a date was given with no time of day."
        ),
    )
    # A Literal, not a str with the options listed in the description. Asked
    # for a free string, the model returned "procedure_cleaning" - it had
    # invented a naming convention from the field name, and the value was
    # dropped as off-schema on every call.
    procedure: Literal[
        "cleaning", "extraction", "root_canal", "filling",
        "checkup", "whitening", "braces", NOT_STATED,
    ] | None = Field(None, description='The procedure requested, or null if not stated.')


SYSTEM_PROMPT = """You extract appointment details from dental clinic call transcripts.

The caller may speak English, Tamil, Kannada, Malayalam or Hindi, and may mix \
an Indic language with English clinical terms in one sentence.

Rules:
- Return only what the caller actually said. If a field was not stated, return null.
- Never infer a time of day that was not spoken. "tomorrow" with no hour is null \
for appointment_datetime, not 09:00.
- Copy names and numbers character for character. Do not correct, translate or \
reformat them.

Set procedure whenever the caller names a treatment anywhere in the sentence. \
Map the words they use onto the code:
  cleaning    - cleaning, scaling, polish, hygiene
  extraction  - extraction, remove a tooth, pull it out
  root_canal  - root canal, RCT, endo
  filling     - filling, cavity
  checkup     - checkup, consultation, "I have pain", "can you look at it"
  whitening   - whitening, bleaching
  braces      - braces, aligners, orthodontic
Only null if no treatment is mentioned at all. A treatment named in English \
inside an Indic sentence still counts.
"""

USER_PROMPT = """Reference time (the moment of the call): {reference_time}

Transcript:
{transcript}
"""


@dataclass
class LLMExtraction:
    """What the model said, mapped back to real values."""

    patient_name: str | None = None
    phone: str | None = None
    appointment_datetime: str | None = None
    procedure: str | None = None
    model: str = DEFAULT_MODEL
    redacted_sent: str = ""

    def value_for(self, slot_name: str) -> Any:
        return {
            "patient_name": self.patient_name,
            "phone": self.phone,
            "appointment_time": self.appointment_datetime,
            "procedure": self.procedure,
        }.get(slot_name)


def build_chat_model(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> Any:
    """Construct the LangChain chat model.

    Imported lazily so the package still imports - and the whole suite still
    runs - on a machine with no langchain-ollama installed. The cross-check
    is an optional enhancement; making it an import-time hard dependency
    would let it take down booking.
    """
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model,
        base_url=base_url,
        # Deterministic: a second opinion that changes between runs cannot be
        # reasoned about, and disagreement would be noise rather than signal.
        temperature=0.0,
        client_kwargs={"timeout": timeout_s},
    )


def extract_llm_slots(
    transcript: str,
    reference_time: Any,
    name: str | None = None,
    phone: str | None = None,
    chat_model: Any = None,
) -> LLMExtraction | None:
    """Extract slots from a redacted transcript. None on any failure.

    ``name`` is the rule extractor's normalised value and ``phone`` its raw
    span - the forms that appear verbatim in the transcript. See ``redact``
    for why those two differ.
    """
    redaction = redact(transcript, name=name, phone=phone)

    # An identifier we were asked to remove and could not find. Nothing
    # downstream will catch a name, so this is the only thing standing
    # between a bad span and a real patient's name on a third party's server.
    if redaction.unremoved:
        log.warning(
            "skipping LLM cross-check: %d identifier(s) could not be redacted",
            len(redaction.unremoved),
        )
        return None

    # Last line of defence. If a digit run survived redaction it is a number
    # the rule extractor did not find - a second number in the utterance, or a
    # format the regex missed - and sending it would put a real mobile on a
    # third party's server. Refusing costs a cross-check; sending cannot be
    # undone.
    leaks = residual_identifiers(redaction.text, allowed=set(redaction.reverse))
    if leaks:
        log.warning(
            "skipping LLM cross-check: %d unredacted identifier-like span(s)", len(leaks)
        )
        return None

    try:
        model = chat_model if chat_model is not None else build_chat_model()
        structured = model.with_structured_output(
            ExtractedSlots, method=STRUCTURED_OUTPUT_METHOD
        )
        result = structured.invoke([
            ("system", SYSTEM_PROMPT),
            ("user", USER_PROMPT.format(
                reference_time=reference_time, transcript=redaction.text,
            )),
        ])
    except Exception as exc:
        # Timeout, connection refused, model not pulled, malformed structured
        # output. All of them mean "no second opinion", never "stop booking".
        log.warning("LLM cross-check unavailable: %s: %s", type(exc).__name__, exc)
        return None

    if result is None:
        return None

    # Sentinels back to None at the boundary, so nothing downstream has to
    # know that "not stated" travels as "" and "none" over the wire.
    procedure = (result.procedure or "").strip().lower()
    if procedure in (NOT_STATED, ""):
        procedure = None
    elif procedure not in PROCEDURE_CODES:
        # Unreachable while the schema is a Literal, kept because the failure
        # it guards - a procedure the model invented treated as evidence about
        # what the caller said - is worse than a redundant check.
        log.warning("LLM returned off-schema procedure %r, ignoring", procedure)
        procedure = None

    return LLMExtraction(
        patient_name=redaction.restore((result.patient_name or "").strip() or None),
        phone=redaction.restore((result.phone or "").strip() or None),
        appointment_datetime=((result.appointment_datetime or "").strip() or None),
        procedure=procedure,
        redacted_sent=redaction.text,
    )
