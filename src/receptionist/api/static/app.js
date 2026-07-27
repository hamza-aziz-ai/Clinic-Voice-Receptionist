"use strict";
/* Transport only. Every figure shown here came from the server.

   The chat is a rendering of session.transcript, not a second source of truth:
   the server owns the conversation and this file redraws it. That matters
   because the transcript is what the read-back gate acts on - a message that
   exists only in the browser would be a message the booking logic never saw. */

const $ = (id) => document.getElementById(id);
const live = (msg) => { $("live-region").textContent = msg; };
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

let callId = null;
let rendered = 0;          // how many transcript turns are already on screen
let lastAudio = null;      // most recent agent audio, for the replay button

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
  tab.focus();
}
tabs.forEach((tab, i) => {
  tab.addEventListener("click", () => selectTab(tab));
  tab.addEventListener("keydown", (e) => {
    const next = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
    if (!next) return;
    e.preventDefault();
    selectTab(tabs[(i + next + tabs.length) % tabs.length]);
  });
});

/* ---------------- chat ----------------
   Messages are appended rather than the list being rebuilt, so the typewriter
   reveal is not restarted every time a booking or message table refreshes. */
function bubble(turn) {
  const li = document.createElement("li");
  li.className = `msg ${turn.speaker === "agent" ? "agent" : "caller"}`;

  const who = document.createElement("span");
  who.className = "visually-hidden";
  who.textContent = turn.speaker === "agent" ? "Agent said: " : "You said: ";

  const body = document.createElement("span");
  body.className = "bubble";

  const text = document.createElement("span");
  text.className = "text";

  body.append(text);
  li.append(who, body);

  if (turn.note) {
    const note = document.createElement("span");
    note.className = "note";
    note.textContent = turn.note;
    li.append(note);
  }
  return { li, text };
}

/* Reveal an agent line a few characters at a time.

   Purely visual: the element is aria-hidden while it fills, and the finished
   sentence is written into a visually-hidden node afterwards so the live
   region announces it once instead of on every frame. A screen reader user
   hearing "W… Wh… Wha…" would be worse served than one who waits. */
function typeInto(node, str) {
  if (REDUCED_MOTION || !str) { node.textContent = str; return Promise.resolve(); }
  node.setAttribute("aria-hidden", "true");
  return new Promise((resolve) => {
    let i = 0;
    const step = Math.max(1, Math.round(str.length / 60));
    const timer = setInterval(() => {
      i = Math.min(str.length, i + step);
      node.textContent = str.slice(0, i);
      scrollChat();
      if (i >= str.length) {
        clearInterval(timer);
        node.removeAttribute("aria-hidden");
        resolve();
      }
    }, 16);
  });
}

function scrollChat() {
  const chat = $("chat");
  chat.scrollTop = chat.scrollHeight;
}

function showTyping() {
  if ($("typing")) return;
  const li = document.createElement("li");
  li.className = "msg agent typing";
  li.id = "typing";
  // The dots are decorative; the label is what assistive tech reads.
  li.innerHTML =
    '<span class="bubble"><span class="dots" aria-hidden="true">' +
    "<i></i><i></i><i></i></span>" +
    '<span class="visually-hidden">The agent is typing</span></span>';
  $("chat").append(li);
  scrollChat();
}

function hideTyping() {
  const el = $("typing");
  if (el) el.remove();
}

/* Draw any transcript turns the screen has not shown yet. */
async function renderChat(session, { animate = true } = {}) {
  const chat = $("chat");
  const turns = (session && session.transcript) || [];

  if (!turns.length) {
    chat.innerHTML = '<li class="empty">No call in progress. Say something, or press New call.</li>';
    rendered = 0;
    return;
  }
  if (rendered === 0) chat.innerHTML = "";

  for (const turn of turns.slice(rendered)) {
    const { li, text } = bubble(turn);
    chat.append(li);
    scrollChat();
    if (animate && turn.speaker === "agent") {
      await typeInto(text, turn.text);
    } else {
      text.textContent = turn.text;
    }
  }
  rendered = turns.length;
  scrollChat();
}

function renderState(session) {
  const state = (session && session.state) || "ready";
  $("chat-state").textContent = {
    greeting: "Ringing…", detect_language: "Listening",
    collect: "Collecting details", confirm: "Confirming",
    book: "Booking", notify: "Sending confirmation",
    ended: "Call ended", escalated: "Handed to reception",
  }[state] || state.replace(/_/g, " ");
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

/* ---------------- audio ---------------- */
function play(base64) {
  if (!base64) return;
  lastAudio = base64;
  // Follows a user gesture (send or mic release), so autoplay policy allows it.
  new Audio(`data:audio/wav;base64,${base64}`).play().catch(() => {});
}

/* ---------------- other panels ---------------- */
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
  rendered = 0;
  await renderChat(s, { animate: false });
  renderSlots(s); renderState(s);
  live("New call started. The agent has greeted the caller.");
}

async function afterTurn(result) {
  hideTyping();
  await renderChat(result);
  renderSlots(result); renderState(result);
  await Promise.all([refreshBookings(), refreshMessages()]);
}

$("call-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const box = $("utterance");
  const err = $("utterance-error");
  const text = box.value.trim();

  if (!text) {
    err.textContent = "Type what the caller said, or hold the microphone to speak.";
    err.hidden = false;
    box.setAttribute("aria-invalid", "true");
    box.focus();
    live("Error: nothing to send.");
    return;
  }
  err.hidden = true;
  box.removeAttribute("aria-invalid");

  if (!callId) await startCall();
  box.value = "";
  autoGrow();
  showTyping();

  let result;
  try {
    result = await api(`/calls/${callId}/utterance`, {
      method: "POST", body: JSON.stringify({ text, now: new Date().toISOString() }),
    });
  } catch (error) {
    hideTyping();
    live(`Could not reach the agent: ${error.message}`);
    return;
  }
  await afterTurn(result);
  live(`Agent replied. Call state: ${result.state.replace(/_/g, " ")}.`);
});

/* ---------------- push to talk ----------------
   Hold to record, release to send. Not continuous listening: deciding when a
   caller has stopped speaking is its own hard problem, and doing it badly
   makes the whole system feel broken for reasons unrelated to whether the
   booking logic is right. The button sidesteps it rather than half-solving it.

   Only the caller is ever recorded - the agent's reply plays through the
   speaker and never enters the microphone stream. That is what makes this
   mono capture safe to score confidence against. */
let recorder = null, chunks = [];

async function startRecording() {
  if (recorder) return;
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
  } catch (err) {
    live(`Microphone unavailable: ${err.message}`);
    return;
  }
  chunks = [];
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  recorder.onstop = () => stream.getTracks().forEach((t) => t.stop());
  recorder.start();
  $("talk").setAttribute("aria-pressed", "true");
  $("talk").classList.add("recording");
  live("Recording. Release to send.");
}

async function stopRecording() {
  if (!recorder) return;
  const done = new Promise((r) => recorder.addEventListener("stop", r, { once: true }));
  recorder.stop();
  await done;
  recorder = null;
  $("talk").setAttribute("aria-pressed", "false");
  $("talk").classList.remove("recording");
  if (!chunks.length) return;

  if (!callId) await startCall();
  showTyping();
  live("Transcribing…");

  const form = new FormData();
  form.append("audio", new Blob(chunks, { type: "audio/webm" }), "turn.webm");

  let result;
  try {
    const res = await fetch(`/calls/${callId}/audio`, { method: "POST", body: form });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    result = await res.json();
  } catch (err) {
    hideTyping();
    live(`Voice failed: ${err.message}. Is the server running with VOICE_ENABLED=1?`);
    return;
  }

  if (!result.heard) {
    hideTyping();
    live("Nothing intelligible was heard. Try again, closer to the microphone.");
    return;
  }
  play(result.audio);
  await afterTurn(result);
  live(`Heard: "${result.heard}". Agent replied${result.audio ? " aloud" : " in text only"}.`);
}

const talk = $("talk");
talk.addEventListener("mousedown", startRecording);
talk.addEventListener("mouseup", stopRecording);
talk.addEventListener("mouseleave", () => { if (recorder) stopRecording(); });
talk.addEventListener("touchstart", (e) => { e.preventDefault(); startRecording(); });
talk.addEventListener("touchend", (e) => { e.preventDefault(); stopRecording(); });
// Keyboard equivalent: a hold gesture that only works with a mouse is not a
// control, it is an obstacle.
talk.addEventListener("keydown", (e) => {
  if ((e.key === " " || e.key === "Enter") && !e.repeat) { e.preventDefault(); startRecording(); }
});
talk.addEventListener("keyup", (e) => {
  if (e.key === " " || e.key === "Enter") { e.preventDefault(); stopRecording(); }
});

/* ---------------- composer ---------------- */
function autoGrow() {
  const box = $("utterance");
  box.style.height = "auto";
  box.style.height = `${Math.min(box.scrollHeight, 160)}px`;
}
$("utterance").addEventListener("input", autoGrow);
$("utterance").addEventListener("keydown", (e) => {
  // Enter sends, Shift+Enter makes a new line - what every chat client does.
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("call-form").requestSubmit();
  }
});

$("new-call").addEventListener("click", async () => { callId = null; await startCall(); });
$("dispatch").addEventListener("click", runDispatch);
document.querySelectorAll(".chip").forEach((c) =>
  c.addEventListener("click", () => {
    $("utterance").value = c.dataset.fill;
    autoGrow();
    $("utterance").focus();
  }));

(async function init() {
  const h = await api("/health");
  $("clinic-status").textContent =
    `${h.chairs} chairs · ${h.languages.length} languages · ${h.active_bookings} active bookings`;
  await startCall();
  await Promise.all([refreshBookings(), refreshMessages(), refreshEval()]);
  live("Console ready.");
})();
