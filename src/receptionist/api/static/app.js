"use strict";
/* Transport only. Every figure shown here came from the server. */

const $ = (id) => document.getElementById(id);
const live = (msg) => { $("live-region").textContent = msg; };
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

let callId = null;

async function api(path, options) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" }, ...options,
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

/* ---------------- tabs ---------------- */
const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
function selectTab(tab) {
  tabs.forEach((t) => {
    const selected = t === tab;
    t.setAttribute("aria-selected", String(selected));
    $(t.getAttribute("aria-controls")).hidden = !selected;
  });
  live(`${tab.textContent} panel shown`);
}
tabs.forEach((tab) => {
  tab.addEventListener("click", () => selectTab(tab));
  tab.addEventListener("keydown", (e) => {
    const i = tabs.indexOf(tab);
    if (e.key === "ArrowRight") { tabs[(i + 1) % tabs.length].focus(); e.preventDefault(); }
    if (e.key === "ArrowLeft")  { tabs[(i - 1 + tabs.length) % tabs.length].focus(); e.preventDefault(); }
  });
});

/* ---------------- rendering ---------------- */
function renderTranscript(session) {
  const el = $("transcript");
  if (!session || !session.transcript.length) {
    el.innerHTML = '<li class="empty">No call in progress.</li>';
    return;
  }
  el.innerHTML = session.transcript.map((t) => `
    <li class="${t.speaker === "agent" ? "agent" : "caller"}">
      <span class="who">${t.speaker === "agent" ? "Agent" : "Caller"}:</span>
      <span>${esc(t.text)}</span>
      ${t.note ? `<span class="note">${esc(t.note)}</span>` : ""}
    </li>`).join("");
}

function renderSlots(session) {
  const el = $("slots");
  if (!session) { el.innerHTML = '<tr><td colspan="4" class="empty">No details captured yet.</td></tr>'; return; }
  el.innerHTML = session.slots.map((s) => {
    let status, cls;
    if (!s.value)                  { status = "Not captured";        cls = "status-warn"; }
    else if (s.confirmed)          { status = "Confirmed by caller"; cls = "status-ok"; }
    else if (s.needs_confirmation) { status = "Read-back required";  cls = "status-warn"; }
    else                           { status = "Accepted";            cls = "status-ok"; }
    const pct = Math.round(s.confidence * 100);
    // The bar is decoration over a number that is already there, so it is
    // hidden from assistive tech rather than duplicated as an ARIA meter.
    // Tinting it when the slot is below threshold makes the confidence and
    // status columns agree at a glance instead of needing to be cross-read.
    const below = s.needs_confirmation ? " below" : "";
    return `<tr>
      <th scope="row">${esc(s.name.replace(/_/g, " "))}</th>
      <td>${esc(s.value ?? "—")}${s.notes.length ? `<span class="note">${esc(s.notes.join("; "))}</span>` : ""}</td>
      <td class="num">
        <span class="meter">
          <span class="meter-track" aria-hidden="true"><span class="meter-fill${below}" style="width:${pct}%"></span></span>
          <span>${pct}%</span>
        </span>
      </td>
      <td><span class="pill ${cls}">${status}</span></td>
    </tr>`;
  }).join("");
}

async function refreshBookings() {
  const rows = await api("/bookings");
  $("bookings").innerHTML = rows.length ? rows.map((b) => `<tr>
      <th scope="row">${esc(new Date(b.start).toLocaleString())}</th>
      <td>${esc(b.patient_name)}</td>
      <td>${esc(b.procedure.replace(/_/g, " "))}</td>
      <td class="num">${b.duration_min} min</td>
      <td>${esc(b.language_name)}</td>
      <td class="num">${esc(b.phone)}</td></tr>`).join("")
    : '<tr><td colspan="6" class="empty">No bookings yet.</td></tr>';
}

async function refreshMessages() {
  const rows = await api("/messages");
  $("messages").innerHTML = rows.length ? rows.map((m) => `<tr>
      <th scope="row">${esc(m.template.replace(/_/g, " "))}</th>
      <td>${esc(m.to)}</td>
      <td>${esc(m.language_name)}</td>
      <td class="num">${m.send_after ? esc(new Date(m.send_after).toLocaleString()) : "immediately"}</td>
      <td><span class="pill ${statusClass(m.status)}">${esc(m.status)}</span></td>
      <td lang="${esc(m.language)}">${esc(m.body)}</td></tr>`).join("")
    : '<tr><td colspan="6" class="empty">No messages queued.</td></tr>';
}

// Expired and failed are errors, not warnings: an expired message means the
// dispatch schedule did not run in time, and nobody was told anything.
function statusClass(status) {
  if (status === "sent") return "status-ok";
  if (status === "expired" || status === "failed") return "status-error";
  return "status-warn";
}

async function runDispatch() {
  const at = $("dispatch-at").value;
  const report = await api(`/messages/dispatch${at ? `?now=${encodeURIComponent(at)}` : ""}`,
                           { method: "POST" });
  await refreshMessages();
  live(
    `Dispatch complete. ${report.sent} sent, ${report.cancelled} cancelled, ` +
    `${report.expired} expired, ${report.failed} failed, ${report.retrying} retrying.`
  );
}

async function refreshEval() {
  const data = await api("/evaluation");
  $("eval").innerHTML = data.rows.map((r) => {
    const silent = r.wrong_silent;
    return `<tr>
      <th scope="row" class="num">${Math.round(r.asr_error_rate * 100)}%</th>
      <td class="num">${(r.slot_accuracy * 100).toFixed(1)}%</td>
      <td class="num">${(r.language_accuracy * 100).toFixed(1)}%</td>
      <td class="num">${r.wrong_caught}</td>
      <td><span class="pill ${silent === 0 ? "status-ok" : "status-error"}">${silent === 0 ? "0 — none" : silent}</span></td>
    </tr>`;
  }).join("");
}

/* ---------------- call handling ---------------- */
async function startCall() {
  const s = await api("/calls", { method: "POST", body: JSON.stringify({ caller_number: "+971500000000" }) });
  callId = s.call_id;
  renderTranscript(s); renderSlots(s);
  live("New call started. The agent has greeted the caller.");
}

$("call-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const box = $("utterance");
  const err = $("utterance-error");
  const text = box.value.trim();

  if (!text) {
    err.textContent = "Enter what the caller said before sending.";
    err.hidden = false;
    box.setAttribute("aria-invalid", "true");
    box.focus();
    live("Error: the utterance field is empty.");
    return;
  }
  err.hidden = true;
  box.removeAttribute("aria-invalid");

  if (!callId) await startCall();
  const result = await api(`/calls/${callId}/utterance`, {
    method: "POST", body: JSON.stringify({ text, now: new Date().toISOString() }),
  });
  box.value = "";
  renderTranscript(result); renderSlots(result);
  await Promise.all([refreshBookings(), refreshMessages()]);
  live(`Agent replied. Call state: ${result.state.replace(/_/g, " ")}.`);
});

$("new-call").addEventListener("click", async () => { callId = null; await startCall(); });
$("dispatch").addEventListener("click", runDispatch);
document.querySelectorAll(".chip").forEach((c) =>
  c.addEventListener("click", () => { $("utterance").value = c.dataset.fill; $("utterance").focus(); }));

(async function init() {
  const h = await api("/health");
  $("clinic-status").textContent =
    `${h.chairs} chairs · ${h.languages.length} languages supported · ${h.active_bookings} active bookings`;
  await startCall();
  await Promise.all([refreshBookings(), refreshMessages(), refreshEval()]);
  live("Console ready.");
})();
