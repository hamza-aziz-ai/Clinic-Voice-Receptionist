"""The n8n workflow may route. It may not decide."""
import json
import re
from pathlib import Path

from receptionist.integrations.n8n import (
    OUTCOMES,
    build_workflow,
    to_json,
)
from receptionist.telephony.ingest import Outcome

EXPORTED = Path(__file__).resolve().parent.parent / "integrations" / "n8n" / "clinic-receptionist.json"


def nodes_by_name() -> dict:
    return {n["name"]: n for n in build_workflow()["nodes"]}


def http_nodes() -> list[dict]:
    return [n for n in build_workflow()["nodes"]
            if n["type"] == "n8n-nodes-base.httpRequest"]


def executable_parameters() -> str:
    """Everything in the workflow that drives behaviour, as one string.

    ``notes`` are excluded: they are documentation shown in the n8n editor,
    and a note explaining that the confidence gate lives in the service is
    the opposite of a violation. Scanning them would make the rule below
    unfollowable - you could not write down what the workflow may not do.
    """
    stripped = []
    for node in build_workflow()["nodes"]:
        params = {k: v for k, v in node["parameters"].items() if k != "notes"}
        stripped.append(params)
    return json.dumps(stripped).lower()


# ------------------------------------------------------------- no decisions
def test_no_node_can_create_a_booking():
    """The only booking path is the service's own gate."""
    blob = executable_parameters()
    for forbidden in ("/bookings", "/calls", "calendar", "book("):
        assert forbidden not in blob, f"workflow reaches for {forbidden!r}"


def test_no_node_branches_on_a_confidence_value():
    """A threshold in a Switch node makes every guarantee here decorative."""
    blob = executable_parameters()
    for forbidden in ("confidence", "threshold", "needs_confirmation", "slots["):
        assert forbidden not in blob, f"workflow inspects {forbidden!r}"


def test_the_service_is_the_only_host_called():
    for node in http_nodes():
        url = node["parameters"]["url"]
        assert url.startswith("={{ $env.RECEPTIONIST_URL }}") or url.startswith(
            "={{ $env.OPS_ALERT_URL }}"
        ), url


# ------------------------------------------------------------- routing
def test_every_service_outcome_is_routed():
    """Adding an outcome to the service must break this, not fall through."""
    assert set(OUTCOMES) == set(Outcome.__args__)


def test_switch_has_no_fallback_branch():
    switch = nodes_by_name()["Route Outcome"]
    assert switch["parameters"]["options"]["fallbackOutput"] == "none"


def test_switch_outputs_line_up_with_its_rules():
    workflow = build_workflow()
    rules = workflow["nodes"][2]["parameters"]["rules"]["values"]
    assert [r["outputKey"] for r in rules] == list(OUTCOMES)
    assert len(workflow["connections"]["Route Outcome"]["main"]) == len(OUTCOMES)


def test_callback_and_escalation_both_reach_a_human():
    routes = build_workflow()["connections"]["Route Outcome"]["main"]
    needs_callback = routes[OUTCOMES.index("needs_callback")][0]["node"]
    escalated = routes[OUTCOMES.index("escalated")][0]["node"]
    assert needs_callback == escalated == "Alert Reception"


def test_every_branch_acknowledges_the_webhook():
    """Bolna retries on non-2xx; an unanswered branch loops forever."""
    connections = build_workflow()["connections"]
    for branch in ("Booked", "Alert Reception", "Ignore Incomplete Call"):
        assert connections[branch]["main"][0][0]["node"] == "Acknowledge"


def test_dispatch_runs_on_a_schedule():
    nodes = nodes_by_name()
    assert nodes["Every 15 Minutes"]["type"] == "n8n-nodes-base.scheduleTrigger"
    dispatch = nodes["Dispatch Due Messages"]["parameters"]
    assert dispatch["url"].endswith("/messages/dispatch")
    assert dispatch["method"] == "POST"


def test_expired_messages_raise_an_alert():
    """Expired means the schedule did not run, which is an operational fault."""
    condition = nodes_by_name()["Alert On Failed Sends"]["parameters"]["conditions"]
    assert "expired" in condition["conditions"][0]["leftValue"]


# ------------------------------------------------------------- secrets
def test_no_secret_is_baked_into_the_exported_json():
    blob = to_json()
    assert "$env.BOLNA_WEBHOOK_SECRET" in blob
    # Anything that looks like a real key rather than an env reference.
    assert not re.search(r'"(apiKey|api_key)"\s*:\s*"[^"{]', blob)


def test_ingest_forwards_the_shared_secret():
    headers = nodes_by_name()["Ingest Call"]["parameters"]["headerParameters"]
    names = {h["name"] for h in headers["parameters"]}
    assert "X-Webhook-Secret" in names


def test_calls_into_the_service_retry():
    for node in http_nodes():
        assert node["parameters"]["options"]["retry"]["retryOnFail"] is True


# ------------------------------------------------------------- export
def test_committed_export_matches_the_generator():
    """Regenerate with: python scripts/export_n8n.py"""
    assert EXPORTED.exists(), f"{EXPORTED} missing - run scripts/export_n8n.py"
    assert EXPORTED.read_text(encoding="utf-8") == to_json()


def test_exported_file_is_valid_json_n8n_can_import():
    data = json.loads(EXPORTED.read_text(encoding="utf-8"))
    assert data["name"] and data["nodes"] and data["connections"]
    for node in data["nodes"]:
        assert {"parameters", "name", "type", "position"} <= set(node)


# ------------------------------------------------- verified against real n8n
# The three defects below were all invisible to schema inspection and were
# found by importing this file into n8n 2.31.7 and POSTing real payloads.
def test_the_workflow_carries_an_id():
    """`n8n import:workflow` fails on a NOT NULL constraint against
    workflow_entity.id without one. Stable, so a re-import updates the
    workflow rather than creating a duplicate beside it."""
    workflow = build_workflow()
    assert workflow["id"]
    assert workflow["id"] == json.loads(EXPORTED.read_text(encoding="utf-8"))["id"]


def test_the_webhook_node_carries_a_webhook_id():
    """n8n keys the registered production URL on webhookId. Without one the
    workflow activates, reports itself active, and every POST to its path
    returns "the requested webhook is not registered"."""
    webhook = nodes_by_name()["Bolna Post-Call"]
    assert webhook.get("webhookId"), "no webhookId - the path will never bind"


def test_the_alert_node_cannot_swallow_the_acknowledgement():
    """Observed live: when the ops alert errored, the run stopped there, the
    Respond node never fired, and the webhook returned an empty body - so
    Bolna would retry a call the service had correctly declined, forever."""
    alert = nodes_by_name()["Alert Reception"]
    assert alert.get("onError") == "continueRegularOutput"


def test_env_access_requirement_is_documented_in_the_workflow():
    """n8n blocks $env in expressions by default, which silently resolves
    every URL in this workflow to undefined. The workflow is useless without
    the operator knowing that, so the requirement travels with the file."""
    blob = to_json()
    assert "$env." in blob
    assert "N8N_BLOCK_ENV_ACCESS_IN_NODE" in blob, (
        "the deployment requirement must be stated in the workflow itself"
    )


def test_calls_into_the_service_have_an_explicit_timeout():
    """Ingest replays every caller turn through the gate, and with LLM
    understanding on each turn is a model call. Left to n8n's default the
    request is cut off mid-booking and retried against a half-processed
    call."""
    for node in http_nodes():
        options = node["parameters"]["options"]
        assert options.get("timeout"), f"{node['name']} has no timeout"

    ingest = nodes_by_name()["Ingest Call"]["parameters"]["options"]["timeout"]
    dispatch = nodes_by_name()["Dispatch Due Messages"]["parameters"]["options"]["timeout"]
    # Ingest is the slow one; dispatch has no model in it.
    assert ingest > dispatch


def test_the_setup_note_separates_n8n_config_from_service_config():
    """Two environments, and confusing them is how someone sets the model in
    n8n and wonders why nothing changed."""
    setup = build_workflow()["meta"]["setup"]
    assert ".env" in setup
    assert "N8N_BLOCK_ENV_ACCESS_IN_NODE" in setup
