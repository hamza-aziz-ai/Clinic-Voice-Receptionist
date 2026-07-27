"""HTTP surface and console host.

Transport only. Every value the console displays was computed in the NLU,
scheduling or evaluation layers; nothing is derived in the browser.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import settings
from ..evaluation.corpus import CASES, REFERENCE
from ..evaluation.harness import evaluate
from ..messaging.aisensy import AiSensyConnector, MockAiSensy
from ..messaging.base import OutboundMessage, render_template
from ..messaging.dispatch import dispatch_due
from ..nlu.language import LANGUAGE_NAMES
from ..scheduling.calendar import Calendar, ClinicHours
from ..telephony.bolna import BolnaWebhookError, parse_execution, verify_source
from ..telephony.ingest import ingest_execution
from ..workflow.call import CallHandler, CallSession

STATIC = Path(__file__).parent / "static"

app = FastAPI(
    title="Clinic Voice Receptionist",
    version="0.1.0",
    description=(
        "Multilingual inbound voice agent for dental clinics. Confidence-gated "
        "slot extraction, transactional booking, WhatsApp follow-up."
    ),
)

calendar = Calendar(hours=ClinicHours(), chairs=2)
# The mock builds the same AiSensy request body the live connector posts, so
# a parameter-order or missing-campaign mistake fails here and in the demo,
# not only against a real account nobody running this repository has.
messaging = (
    AiSensyConnector(settings.aisensy_api_key, settings.aisensy_base_url)
    if settings.aisensy_api_key
    else MockAiSensy()
)
def _crosscheck_hook():
    """The LLM second opinion, or None when it is switched off.

    Building the chat model once at import would make a cold Ollama a
    startup failure for the whole API. It is constructed on first use inside
    the extractor instead, and every error there returns None.
    """
    if not settings.llm_crosscheck_enabled:
        return None

    from ..nlu.llm_extractor import build_chat_model, extract_llm_slots

    model = None

    def hook(text, now, name, phone):
        nonlocal model
        if model is None:
            model = build_chat_model(settings.llm_model, settings.llm_base_url)
        return extract_llm_slots(text, now, name, phone, chat_model=model)

    return hook


handler = CallHandler(
    calendar, messaging,
    clinic_name=settings.clinic_name, review_link=settings.review_link,
    crosscheck=_crosscheck_hook(),
)
SESSIONS: dict[str, CallSession] = {}


class StartCall(BaseModel):
    caller_number: str = Field("", examples=["+971501112222"])


class Utterance(BaseModel):
    text: str
    word_confidences: dict[str, float] | None = None
    now: datetime | None = None


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "languages": [c for c in LANGUAGE_NAMES if c != "uncertain"],
        "chairs": calendar.chairs,
        "active_bookings": len(calendar.active()),
        "calls": len(SESSIONS),
    }


@app.post("/calls")
def start_call(req: StartCall) -> dict[str, Any]:
    session = handler.start(req.caller_number)
    SESSIONS[session.call_id] = session
    return session.to_dict()


@app.post("/calls/{call_id}/utterance")
def utterance(call_id: str, req: Utterance) -> dict[str, Any]:
    session = SESSIONS.get(call_id)
    if session is None:
        raise HTTPException(404, f"No call {call_id!r}")
    reply = handler.handle_utterance(
        session, req.text, req.now or datetime.now(), req.word_confidences
    )
    return {"reply": reply, **session.to_dict()}


@app.get("/calls")
def list_calls() -> list[dict[str, Any]]:
    return [s.to_dict() for s in SESSIONS.values()]


@app.get("/calls/{call_id}")
def get_call(call_id: str) -> dict[str, Any]:
    session = SESSIONS.get(call_id)
    if session is None:
        raise HTTPException(404, f"No call {call_id!r}")
    return session.to_dict()


@app.get("/bookings")
def bookings() -> list[dict[str, Any]]:
    return [
        {
            "booking_id": b.booking_id, "patient_name": b.patient_name,
            "phone": b.phone, "procedure": b.procedure,
            "start": b.start.isoformat(), "end": b.end.isoformat(),
            "duration_min": b.duration_min, "language": b.language,
            "language_name": LANGUAGE_NAMES.get(b.language, b.language),
            "status": b.status,
        }
        for b in sorted(calendar.active(), key=lambda x: x.start)
    ]


@app.get("/messages")
def messages() -> list[dict[str, Any]]:
    out = []
    for s in SESSIONS.values():
        for m in s.messages:
            out.append({
                "template": m.template, "to": m.to, "language": m.language,
                "language_name": LANGUAGE_NAMES.get(m.language, m.language),
                "status": m.status,
                "send_after": m.send_after.isoformat() if m.send_after else None,
                "booking_id": m.booking_id,
                "body": render_template(m.template, m.language, m.parameters),
            })
    return out


@app.post("/webhooks/bolna")
async def bolna_webhook(
    request: Request,
    x_webhook_secret: str = Header(default=""),
) -> dict[str, Any]:
    """Post-call execution delivered by Bolna.

    Returns 200 for anything successfully interpreted, including calls that
    deliberately did not book. Bolna retries on non-2xx, and a call the
    system correctly declined to book on is not a delivery failure - answering
    503 to it would produce an endless redelivery loop over a payload that
    will never book no matter how many times it arrives.
    """
    try:
        verify_source(
            remote_ip=request.client.host if request.client else "",
            provided_secret=x_webhook_secret,
            expected_secret=settings.bolna_webhook_secret,
            allowed_ips=settings.bolna_allowed_ips,
        )
        execution = parse_execution(
            await request.json(), settings.clinic_utc_offset_hours
        )
    except BolnaWebhookError as exc:
        raise HTTPException(403, str(exc)) from exc
    except ValueError as exc:                       # malformed JSON body
        raise HTTPException(400, f"unreadable payload: {exc}") from exc

    result = ingest_execution(execution, handler)
    if result.session is not None:
        SESSIONS[result.session.call_id] = result.session
    return result.to_dict()


def _all_messages() -> list[OutboundMessage]:
    return [m for s in SESSIONS.values() for m in s.messages]


@app.post("/messages/dispatch")
def dispatch(now: datetime | None = None) -> dict[str, Any]:
    """Run one pass over the scheduled queue.

    Exposed as an endpoint rather than run on a background thread so the
    scheduler is something you can point at, trigger and observe. n8n's Cron
    node calls this; the console has a button for it; the tests call it with
    an explicit ``now`` instead of waiting a day for a reminder to come due.
    """
    return dispatch_due(_all_messages(), messaging, calendar, now).to_dict()


@app.get("/evaluation")
def evaluation() -> dict[str, Any]:
    rows = []
    for severity in (0.0, 0.15, 0.30, 0.50):
        r = evaluate(CASES, REFERENCE, severity, seed=11)
        rows.append({
            "asr_error_rate": severity,
            "slot_accuracy": round(r.slot_accuracy, 4),
            "language_accuracy": round(r.language_accuracy, 4),
            "silent_error_rate": round(r.silent_error_rate, 4),
            **r.by_outcome(),
        })
    return {
        "cases": len(CASES),
        "slots_checked": len(evaluate(CASES, REFERENCE, 0.0).outcomes),
        "rows": rows,
    }


app.mount("/console", StaticFiles(directory=STATIC, html=True), name="console")


@app.get("/")
def root() -> RedirectResponse:
    """Redirect rather than serve the same file from two paths.

    Serving index.html here looks equivalent and is not: its stylesheet and
    script are relative, so from "/" they resolve to /styles.css and /app.js,
    which are not mounted. The page rendered unstyled and inert, and the
    tests missed it because fetching "/" and fetching /console/app.js both
    returned 200 - just never from the same document.
    """
    return RedirectResponse("/console/")
