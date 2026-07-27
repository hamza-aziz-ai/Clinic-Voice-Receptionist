"""AiSensy WhatsApp connector.

THE POSITIONAL PARAMETER PROBLEM

AiSensy's campaign API takes ``templateParams`` as a **positional array of
strings**, and the array length must equal the number of placeholders in the
approved campaign. Our templates use named placeholders - ``{name}``,
``{procedure}``, ``{when}`` - because named ones are readable and cannot be
transposed by editing.

Converting named to positional is where this integration can fail in the
worst available way. Swap two entries and the API accepts it, WhatsApp
delivers it, and the patient reads "Hello cleaning, your Priya Menon at Al
Noor Dental is confirmed". Nothing errors. Nothing retries. There is no
status code for "grammatically valid nonsense".

So the ordering is declared once, per template, in ``PARAM_ORDER``, next to
the template it belongs to; the conversion reads that list rather than
relying on dict iteration order; and a length or key mismatch raises before
anything is sent. A campaign whose approved placeholder count has drifted
from ours fails loudly at the first send rather than mangling every message
quietly.

WHY ONE CAMPAIGN PER LANGUAGE

WhatsApp approves message templates per language, so "confirmation in Tamil"
and "confirmation in Hindi" are separate approved templates and therefore
separate AiSensy campaigns. The registry is explicit for the same reason the
template bodies are: a missing entry should be a visible gap, not a silent
fallback to whichever campaign happened to be Live.
"""
from __future__ import annotations

from typing import Any

from .base import TEMPLATES, MessagingConnector, OutboundMessage, TemplateName

# Order in which named parameters are flattened into AiSensy's positional
# templateParams array. Must match the placeholder order of the approved
# WhatsApp template for the campaign, which is why it lives beside the
# template bodies rather than at the call site.
PARAM_ORDER: dict[TemplateName, tuple[str, ...]] = {
    "appointment_confirmation": ("name", "procedure", "clinic", "when"),
    "appointment_reminder": ("name", "procedure", "clinic", "when"),
    "review_request": ("clinic", "name", "link"),
}

# (template, language) -> AiSensy campaign name. The campaign must exist and
# be Live in the AiSensy dashboard, wrapping a WhatsApp template Meta has
# approved for that language.
CAMPAIGNS: dict[tuple[str, str], str] = {
    (template, language): f"clinic_{template}_{language}"
    for template in TEMPLATES
    for language in TEMPLATES[template]
}

AISENSY_CAMPAIGN_PATH = "/campaign/t1/api/v2"


class AiSensyError(RuntimeError):
    """Refused before the request left the process."""


def build_request(
    message: OutboundMessage,
    api_key: str,
    source: str = "clinic-voice-receptionist",
) -> dict[str, Any]:
    """Map an OutboundMessage onto the AiSensy campaign request body.

    Raises rather than guessing on any mismatch. Every failure mode here
    produces a *delivered* wrong message if it is papered over, which is
    strictly worse than not sending.
    """
    if not api_key:
        raise AiSensyError("no AiSensy API key configured")

    order = PARAM_ORDER.get(message.template)
    if order is None:
        raise AiSensyError(f"no parameter order declared for template {message.template!r}")

    campaign = CAMPAIGNS.get((message.template, message.language))
    if campaign is None:
        # Falling back to the English campaign would send English text under
        # a Tamil-language contact record, which reads as a bug to the clinic
        # and as a wrong-number to the patient.
        raise AiSensyError(
            f"no approved campaign for {message.template!r} in {message.language!r}"
        )

    missing = [key for key in order if key not in message.parameters]
    if missing:
        raise AiSensyError(
            f"template {message.template!r} needs {', '.join(missing)}"
        )

    return {
        "apiKey": api_key,
        "campaignName": campaign,
        "destination": message.to,
        "userName": message.parameters.get("name", ""),
        "source": source,
        # Positional, in the declared order. Never dict order.
        "templateParams": [str(message.parameters[key]) for key in order],
        "tags": [message.template, f"lang:{message.language}"],
        "attributes": {"booking_id": message.booking_id or ""},
    }


class AiSensyConnector(MessagingConnector):
    """Live connector. Not exercised by the test suite - see MockAiSensy.

    Kept deliberately thin: build the body, post it, classify the response.
    Anything cleverer here would be logic that the mock does not share, and
    the mock is what every test and the demo actually run.
    """

    name = "aisensy"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://backend.aisensy.com",
        timeout_s: float = 10.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def send(self, message: OutboundMessage) -> dict[str, Any]:
        # httpx2, not httpx. Starlette's test client already prefers httpx2
        # when both are importable, and having the app under test on one
        # major version while the connector is on another is how a transport
        # behaviour change - redirect handling, timeout semantics - reaches
        # production without appearing in a single test.
        import httpx2

        body = build_request(message, self.api_key)
        url = f"{self.base_url}{AISENSY_CAMPAIGN_PATH}"
        try:
            response = httpx2.post(url, json=body, timeout=self.timeout_s)
        except httpx2.HTTPError as exc:
            # Transport failure is retryable; a rejected template is not.
            # The dispatcher needs to tell them apart, so say which it was.
            return {"ok": False, "provider": self.name, "retryable": True,
                    "error": f"{type(exc).__name__}: {exc}"}

        ok = response.status_code < 400
        return {
            "ok": ok,
            "provider": self.name,
            "status_code": response.status_code,
            "retryable": response.status_code >= 500 or response.status_code == 429,
            "campaign": body["campaignName"],
            "response": _safe_json(response),
        }


class MockAiSensy(MessagingConnector):
    """Records what would be sent, having built the real request body.

    Building the real body is the point. A mock that skips ``build_request``
    would never catch a parameter-order or missing-campaign mistake, which is
    the class of bug this module exists to prevent.
    """

    name = "aisensy-mock"

    def __init__(self, api_key: str = "mock-key") -> None:
        self.api_key = api_key
        self.sent: list[OutboundMessage] = []
        self.requests: list[dict[str, Any]] = []

    def send(self, message: OutboundMessage) -> dict[str, Any]:
        body = build_request(message, self.api_key)
        self.sent.append(message)
        self.requests.append(body)
        return {
            "ok": True,
            "provider": self.name,
            "campaign": body["campaignName"],
            "templateParams": body["templateParams"],
            "message_id": f"mock-{len(self.sent)}",
        }


def _safe_json(response: Any) -> Any:
    try:
        return response.json()
    except Exception:
        return {"raw": response.text[:500]}
