"""HTTP surface and console host.

Transport only. Every value the console displays was computed in the NLU,
scheduling or evaluation layers; nothing is derived in the browser.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..evaluation.corpus import CASES, REFERENCE
from ..evaluation.harness import evaluate
from ..messaging.base import MockWhatsApp, render_template
from ..nlu.language import LANGUAGE_NAMES
from ..scheduling.calendar import Calendar, ClinicHours
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
messaging = MockWhatsApp()
handler = CallHandler(calendar, messaging)
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
def root() -> FileResponse:
    return FileResponse(STATIC / "index.html")
