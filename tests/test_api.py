from __future__ import annotations
import pytest
from fastapi.testclient import TestClient
from receptionist.api.main import app
from receptionist.config import settings

client = TestClient(app)

def booked_call(at: str) -> str:
    """A call that reaches a booking, at a stated time.

    Each test picks its own time: the app object is module-level, so tests
    share one calendar, and two chairs means the third test asking for the
    same slot would be legitimately refused.
    """
    return (
        "assistant: Thank you for calling.\n"
        f"user: my name is Priya Menon I need a cleaning tomorrow at {at}\n"
        "assistant: I have your name as Priya Menon. Did I get that right?\n"
        "user: yes that's right\n"
        "user: my number is 0501234567\n"
        "assistant: Is that correct?\n"
        "user: yes correct\n"
        "assistant: Shall I book that?\n"
        "user: yes please\n"
    )


def execution(transcript: str, **over) -> dict:
    # created_at is pinned so "tomorrow" always resolves to a Tuesday. Left
    # to the real clock, the whole file fails every Thursday, when tomorrow
    # is the Friday the clinic is closed.
    body = {
        "id": "exec-api-1", "agent_id": "agent-1", "status": "completed",
        "transcript": transcript,
        "created_at": "2026-07-27T06:00:00Z",       # 10:00 Monday, UTC+4
        "telephony_data": {"from_number": "+971501234567", "call_type": "inbound"},
    }
    body.update(over)
    return body


@pytest.fixture
def webhook_open(monkeypatch):
    """TestClient presents as 'testclient', which is not a Bolna address."""
    monkeypatch.setattr(settings, "bolna_webhook_secret", "s3cret")
    monkeypatch.setattr(settings, "bolna_extra_source_ips", ("testclient",))
    return {"X-Webhook-Secret": "s3cret"}


def test_health():
    b = client.get("/health").json()
    assert b["status"] == "ok"
    assert set(b["languages"]) == {"en", "ta", "kn", "ml", "hi"}


def test_call_lifecycle():
    s = client.post("/calls", json={"caller_number": "+971501112222"}).json()
    r = client.post(f"/calls/{s['call_id']}/utterance", json={
        "text": "my name is Priya Menon I need a cleaning",
        "now": "2026-07-27T09:00:00"}).json()
    assert r["reply"]
    assert any(x["name"] == "procedure" and x["value"] == "cleaning" for x in r["slots"])


def test_unknown_call_404():
    assert client.post("/calls/nope/utterance", json={"text": "hi"}).status_code == 404


def test_evaluation_endpoint_reports_zero_silent_errors():
    data = client.get("/evaluation").json()
    assert data["cases"] >= 10
    assert all(row["wrong_silent"] == 0 for row in data["rows"])


def test_webhook_rejects_unconfigured_deployment(monkeypatch):
    """No secret set must not mean no check."""
    monkeypatch.setattr(settings, "bolna_webhook_secret", "")
    r = client.post("/webhooks/bolna", json=execution(booked_call("3 pm")))
    assert r.status_code == 403
    assert "no webhook secret configured" in r.json()["detail"]


def test_webhook_rejects_wrong_secret(webhook_open):
    r = client.post("/webhooks/bolna", json=execution(booked_call("3 pm")),
                    headers={"X-Webhook-Secret": "wrong"})
    assert r.status_code == 403


def test_webhook_rejects_unlisted_source(monkeypatch):
    monkeypatch.setattr(settings, "bolna_webhook_secret", "s3cret")
    monkeypatch.setattr(settings, "bolna_extra_source_ips", ())
    r = client.post("/webhooks/bolna", json=execution(booked_call("3 pm")),
                    headers={"X-Webhook-Secret": "s3cret"})
    assert r.status_code == 403


def test_webhook_books_and_queues_whatsapp(webhook_open):
    body = client.post("/webhooks/bolna",
                       json=execution(booked_call("3 pm"), id="exec-books"),
                       headers=webhook_open).json()
    assert body["outcome"] == "booked"
    assert body["booking_id"]

    booking = next(b for b in client.get("/bookings").json()
                   if b["booking_id"] == body["booking_id"])
    assert booking["patient_name"] == "Priya Menon"

    queued = [m for m in client.get("/messages").json()
              if m["booking_id"] == body["booking_id"]]
    assert {m["template"] for m in queued} == {
        "appointment_confirmation", "appointment_reminder", "review_request"}


def test_webhook_declines_to_book_without_confirmation(webhook_open):
    body = client.post(
        "/webhooks/bolna",
        json=execution("assistant: Hi.\nuser: I need a cleaning tomorrow at 3pm",
                       id="exec-callback"),
        headers=webhook_open,
    ).json()
    assert body["outcome"] == "needs_callback"
    assert body["booking_id"] is None


def test_webhook_redelivery_is_a_200_not_a_duplicate(webhook_open):
    """Bolna retries on non-2xx; a declined call is not a delivery failure."""
    payload = execution(booked_call("4 pm"), id="exec-retry")
    first = client.post("/webhooks/bolna", json=payload, headers=webhook_open)
    second = client.post("/webhooks/bolna", json=payload, headers=webhook_open)
    assert first.status_code == second.status_code == 200
    assert first.json()["outcome"] == "booked"
    assert first.json()["booking_id"] == second.json()["booking_id"]


def test_dispatch_sends_the_reminder_only_once_it_is_due(webhook_open):
    body = client.post("/webhooks/bolna",
                       json=execution(booked_call("5 pm"), id="exec-dispatch"),
                       headers=webhook_open).json()
    assert body["outcome"] == "booked"

    queued = [m for m in client.get("/messages").json()
              if m["booking_id"] == body["booking_id"]]
    reminder = next(m for m in queued if m["template"] == "appointment_reminder")

    # Nothing due yet: the confirmation already went out during the call.
    early = client.post("/messages/dispatch", params={"now": "2026-07-01T00:00:00"}).json()
    assert early["sent"] == 0

    due = client.post("/messages/dispatch",
                      params={"now": reminder["send_after"]}).json()
    assert due["sent"] >= 1
    assert any("appointment_reminder" in label for label in due["detail"]["sent"])


def test_console_is_served():
    assert client.get("/console/").status_code == 200
    assert client.get("/console/styles.css").status_code == 200
    assert client.get("/console/app.js").status_code == 200


def test_root_redirects_rather_than_serving_a_second_copy():
    """Served at "/", index.html's relative assets resolve to /styles.css and
    /app.js, which are not mounted - the page renders unstyled and inert."""
    assert client.get("/", follow_redirects=False).status_code in (307, 308)
    assert client.get("/", follow_redirects=False).headers["location"] == "/console/"


def test_every_asset_the_console_references_resolves():
    """The bug this catches: 200 on the page and 200 on the asset, but never
    from the same document."""
    import re as _re
    html = client.get("/console/").text
    refs = _re.findall(r'(?:href|src)="([^"#:]+)"', html)
    assert refs, "console references no assets at all"
    for ref in refs:
        target = ref if ref.startswith("/") else f"/console/{ref}"
        assert client.get(target).status_code == 200, f"{ref} 404s from /console/"


def test_console_html_has_accessibility_landmarks():
    html = client.get("/console/").text
    for marker in ('lang="en"', "skip-link", 'role="status"', 'aria-live="polite"',
                   "<main", 'scope="col"', "<caption"):
        assert marker in html, f"missing {marker}"


def test_utterance_accepts_an_aware_timestamp_from_the_browser():
    """The console posts new Date().toISOString() - UTC with a trailing Z.
    That reached the extractor as an aware datetime and raised "can't compare
    offset-naive and offset-aware datetimes", so the main Send to agent button
    500'd on every click in a real browser. The suite missed it because these
    tests posted a naive string."""
    s = client.post("/calls", json={"caller_number": "+971501112222"}).json()
    r = client.post(f"/calls/{s['call_id']}/utterance", json={
        "text": "my name is Priya Menon I need a cleaning",
        "now": "2026-07-27T06:00:00Z"})
    assert r.status_code == 200, r.text
    assert any(x["name"] == "procedure" and x["value"] == "cleaning"
               for x in r.json()["slots"])


def test_aware_timestamp_is_converted_not_merely_stripped():
    """17:00Z is 21:00 in Dubai. Dropping the offset instead of converting
    resolves "tomorrow" against the wrong day for any evening caller."""
    s = client.post("/calls", json={}).json()
    body = client.post(f"/calls/{s['call_id']}/utterance", json={
        "text": "my name is Priya Menon a cleaning tomorrow at 3 pm",
        "now": "2026-07-27T17:00:00Z"}).json()
    slot = next(x for x in body["slots"] if x["name"] == "appointment_time")
    # 27 July 21:00 clinic-local, so "tomorrow" is the 28th.
    assert slot["value"].startswith("2026-07-28T15:00")
