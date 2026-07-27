from __future__ import annotations
from fastapi.testclient import TestClient
from receptionist.api.main import app

client = TestClient(app)


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


def test_console_is_served():
    assert client.get("/").status_code == 200
    assert client.get("/console/styles.css").status_code == 200
    assert client.get("/console/app.js").status_code == 200


def test_console_html_has_accessibility_landmarks():
    html = client.get("/").text
    for marker in ('lang="en"', "skip-link", 'role="status"', 'aria-live="polite"',
                   "<main", 'scope="col"', "<caption"):
        assert marker in html, f"missing {marker}"
