# Clinic Voice Receptionist

Multilingual inbound voice agent for dental clinics — English, Tamil, Kannada,
Malayalam and Hindi. Books appointments, sends WhatsApp follow-up, and
**never books on something it isn't sure it heard.**

```
$ python scripts/demo.py       # no API keys, no network, no telephony account
$ python -m pytest -q          # 197 passed
$ uvicorn receptionist.api.main:app --app-dir src   # console at localhost:8000
```

Built against a published brief: **Bolna AI** for the inbound voice agent,
**AiSensy** for WhatsApp, **n8n** as the glue. All three are integrated at the
contract level — real payload shapes, real request bodies, an importable
workflow — behind interfaces with mocks, because I have no accounts and
pretending otherwise would make the tests meaningless.

---

## The problem

A voice receptionist fails differently from a chatbot. The failure isn't "the
model said something wrong" — it's **"the model heard something wrong and
booked on it with complete confidence."**

A misheard digit sends the confirmation WhatsApp to a stranger. A misheard
date books the wrong day. Either way the caller hangs up believing they have
an appointment, and the clinic finds out when nobody arrives.

So the design question isn't "how accurate is the ASR". It's **"what does the
system do when the ASR is wrong, and does it know?"**

---

## Confidence is tracked per slot, not per call

Each field carries its own confidence, derived from things that are actually
measurable: ASR word-level confidence over the span the value came from,
whether the value survived normalisation, and how intrinsically risky the
field is.

| Field | Threshold | Why |
|---|---:|---|
| Phone | 0.92 | One wrong digit is still a *valid* number — nothing downstream can detect it |
| Appointment time | 0.85 | Wrong day is recoverable but expensive |
| Patient name | 0.75 | Proper nouns are out-of-vocabulary for most ASR |
| Procedure | 0.70 | Closed vocabulary — a near-miss still lands on a valid value |

Below threshold, the value is **neither guessed nor discarded** — it is read
back to the caller for explicit confirmation, per field:

```
caller: my name is Priya Menon I need a cleaning tomorrow at 3 pm
        my number is nine seven one five zero one two three four five six seven
agent : I have your name as Priya Menon. Did I get that right?
agent : Let me confirm your number: 9 7 1 5 0 1 2 3 4 5 6 7. Is that correct?
agent : That's Tuesday 28 July at 03:00 PM. Shall I book that?
```

Read-backs are per-slot deliberately. "Did I get all that right?" after a
four-field summary produces a yes that means nothing — a caller cannot hold
four values in mind and check each one.

An ambiguous answer is not treated as yes. Three failed confirmations escalate
to a human, which is better product behaviour than booking something plausible.

---

## Measuring quality in languages I don't speak

I don't speak Tamil, Kannada or Malayalam. On a voice project that isn't a
footnote — it means "it sounded fine" is unavailable to me as a quality signal.

The answer is to stop relying on listening. Every test utterance is paired
with the structured values it must produce, so correctness becomes *"did the
extractor return the ground truth"* — checkable without understanding a word,
and stronger than a fluent speaker skimming transcripts because it's
exhaustive and runs on every commit.

| Simulated ASR error | Slot accuracy | Language accuracy | Caught by read-back | **Silent errors** |
|---:|---:|---:|---:|---:|
| 0% | 100.0% | 100.0% | 0 | **0** |
| 15% | 82.5% | 100.0% | 7 | **0** |
| 30% | 75.0% | 100.0% | 10 | **0** |
| 50% | 75.0% | 100.0% | 10 | **0** |

A **silent error** is a wrong value that was *not* flagged for read-back. It
is the only kind that reaches a patient, and it's the number the system is
designed around. Note what happens as ASR degrades: raw accuracy falls, but
silent errors stay at zero, because degradation lowers confidence, which
triggers read-backs, which catch the errors. **The safety mechanism scales
with the risk.**

---

## Two bugs with the same root cause

Both found by the harness, both invisible to inspection.

**Language detection failed on every code-switched call.** Real calls mix
Indic script with English clinical terms ("എനിക്ക് filling വേണം"). Detection
sat at 60%. Cause: Indic combining vowel signs and the virama are Unicode
category `Mn`/`Mc`, not `Lo` — so `str.isalpha()` returns **False** for them.
Filtering on `isalpha()` before counting discarded roughly a fifth of the
Indic characters in a typical utterance and pushed every mixed call below
threshold. Fixed → 100%.

**The same assumption mangled names.** Python's `\w` doesn't match those marks
either, so the name sanitiser silently rewrote `അഞ്ജലി` as `അഞ ജല`.

The generalisable lesson: **Indic scripts are largely built from marks, and
most Unicode-naive code treats marks as punctuation.** Anything doing
character-class filtering on Indian-language text is probably wrong in a way
that looks fine in English tests.

A third, unrelated: `"15/08 at 11am"` booked at **15:00**, because the time
regex matched the day-of-month before reaching `11am`. Four hours late,
silently.

---

## Booking is transactional

Two calls can arrive in the same second. A check-then-book agent tells both
callers yes.

```
20 concurrent bookings for one chair at 15:00 →  1 succeeded, 19 rejected with alternatives
calendar holds 1 appointment
webhook retry with the same idempotency key → same booking id
```

Plus clinic hours, the Friday closure, lunch break, and per-procedure
durations driving the conflict window (a 90-minute root canal blocks
differently from a 20-minute checkup).

---

## A second extractor that is only allowed to disagree

`gpt-oss:120b-cloud` via Ollama and LangChain, extracting the same four slots
independently. Off by default (`LLM_CROSSCHECK_ENABLED=1`).

**Disagreement lowers confidence. Agreement never raises it.** The symmetric
version — two extractors agree, so trust the value more — is wrong in the
direction that hurts. Both read the *same* degraded transcript, so their errors
are correlated: when ASR turns "five" into "nine", both read nine and both
agree, confidently, on a wrong number. A symmetric rule takes that as evidence
and pushes the slot over its threshold. The worst case of the asymmetric design
is a read-back that was not strictly necessary; the worst case of the symmetric
one is a wrong appointment nobody was asked about.

It also never reports its own confidence and never overwrites a value. A model
asked "how sure are you?" answers with the same process that produced the
answer, so a wrong answer arrives with a wrong confidence attached.

### What it honestly checks

Names and phone numbers are replaced with **surrogates of the same script and
shape** before anything leaves the machine — a Malayalam name becomes a
different Malayalam name, a UAE mobile a different valid UAE mobile — so
ollama.com never sees a patient. Masking them as `<NAME>` instead would have
destroyed the segmentation problem the model is being asked to solve.

The cost is real and worth stating: whatever the model returns for those two
fields is a value **we injected**, so agreement on them is guaranteed and means
nothing. The cross-check has teeth on `appointment_time` and `procedure` only.
Those two are compared; the redacted two are not compared at all, because a
reassuring number that means nothing is worse than no number.

If any identifier cannot be redacted verbatim, or a digit run survives that the
rule extractor never found, **nothing is sent**. That guard is what exposed the
name bug fixed in the commit before this one.

### Measured on the corpus: it does not currently pay for itself

`python scripts/eval_crosscheck.py` — same corpus, same seed, same degraded
transcripts, rules-only versus rules-plus-cross-check.

| ASR error | | Slot accuracy | **Silent errors** | Read-backs | **Rescued** | False alarms |
|---:|---|---:|---:|---:|---:|---:|
| 0% | rules only | 100.0% | **0** | 10 | – | – |
| 0% | + cross-check | 100.0% | **0** | 12 | **0** | 2 |
| 15% | rules only | 83.8% | **0** | 14 | – | – |
| 15% | + cross-check | 83.8% | **0** | 15 | **0** | 1 |
| 30% | rules only | 75.7% | **0** | 19 | – | – |
| 30% | + cross-check | 75.7% | **0** | 20 | **0** | 1 |
| 50% | rules only | 75.7% | **0** | 25 | – | – |
| 50% | + cross-check | 75.7% | **0** | 26 | **0** | 1 |

*40 model calls, 301s, 7.5s each.*

**Rescued is zero at every severity, and the cross-check adds one to two
unnecessary read-backs per run.** On this corpus it is a pure cost.

That is not a surprising result once stated: *rescued* counts wrong values that
would have gone through unflagged, and the silent-error column is already zero
without it. There is nothing to rescue. A safety net under a floor catches
nothing and occasionally trips someone.

So it stays **off by default**, and the honest summary is that it is unproven
rather than useless. The one silent error it has actually caught — `"Sara Ali
root canal"`, in the commit before it — was found by its redaction guard on a
transcript the corpus does not contain, and that bug is now fixed in the rules
where it belongs. The condition under which this feature would earn its place is
a corpus that still contains silent errors; building one means finding classes of
input where the rules are confidently wrong, which is the more valuable work and
is not done here.

The measurement is the deliverable. A feature that ships with a table showing it
currently buys nothing is worth more than one that ships with a plausible story.

### What the model actually did

| Finding | Detail |
|---|---|
| LangChain's default `json_schema` method | **Ignored entirely.** The model answered under invented field names — `caller_name`, `treatment_code` — failing validation on every call. Ollama's `format` is not enforced for this cloud-proxied model. |
| `method="function_calling"` | Correct field names, 5/5 procedure agreement, every datetime right including `15/08 at 11am`. |
| `procedure` as a bare `str` | Returned `procedure_cleaning` — it invented a convention from the field name. A `Literal` fixed it. |
| Latency | **4.8s per call.** Too slow for a conversational turn. |

That last row is a design constraint, not a footnote. The post-call ingest asks
**once for the whole transcript** rather than once per replayed turn, taking a
five-turn call from ~24s of webhook time to ~5s, on a question whose answer does
not change between turns.

The failure mode worth naming: while the schema was being ignored, the model
*had* the right answer and the integration threw it away. **A cross-check that
silently returns nothing looks exactly like a cross-check that agrees.** Every
failure path is logged and surfaces as `available: false` rather than as
silence.

Ollama being down, timing out, or returning garbage all degrade to "no second
opinion" and never to a failed call. A receptionist that stops booking because a
language model is unreachable is worse than one that never had a language model.

---

## Reading the vendor docs changed the design

I assumed Bolna's webhook delivered a transcript with per-word ASR
confidences. It does not. The execution payload carries `transcript` as a
single string — `"assistant: …\nuser: …"` — and no confidence data at all.

That removes the strongest input to the confidence model. Every slot lands on
the no-metadata default, which is below the phone threshold, so **a post-call
payload cannot clear the gate on its own.** Which is correct: nobody is on the
line to be asked.

What rescues the booking is that the read-backs already happened *during* the
call, and the caller's answers are in the transcript. So the caller's turns
are **replayed through the same state machine the console drives**:

```
Bolna execution → caller turns → CallHandler → confidence gate → Calendar.book
```

The tempting alternative was to read `extracted_data` and book what Bolna says
the caller wanted. That would have made everything above decorative — Bolna's
extraction has no per-slot threshold, so the one code path that books real
appointments would bypass the read-back gate entirely. `extracted_data` still
contributes: its `confidence_label` becomes a **ceiling**, and a ceiling can
only lower a slot. A "High" label cannot lift a value over a threshold it
failed on its own merits.

Three outcomes, all valid: `booked`, `needs_callback`, `escalated`. The middle
one is the interesting one — the call happened, the caller wanted an
appointment, and the system is deliberately declining to create one because it
is not sure what it heard. The reason names the specific slots, so reception
knows what to ask instead of replaying the recording. A clinic can work that
queue; it cannot work a calendar full of plausible wrong bookings.

**Bolna also signs nothing.** The documented protection is source-IP
allowlisting. That is not authentication for an endpoint that creates
appointments and messages real numbers, so a shared secret is required as
well — and an unset secret raises rather than skipping the check, because the
fail-open version turns a missing environment variable into a public booking
endpoint. Putting n8n in the middle weakens this further, since the address the
service sees becomes n8n's; that is stated in the module rather than
discovered.

---

## The failure mode WhatsApp cannot report

AiSensy takes `templateParams` as a **positional array of strings**. Our
templates use named placeholders. That conversion is where this integration
fails worst: transpose two entries and the API accepts it, WhatsApp delivers
it, and the patient reads

> Hello **cleaning**, your **Priya Menon** at Al Noor Dental is confirmed…

Nothing errors. Nothing retries. There is no status code for grammatically
valid nonsense. So the ordering is declared once per template, next to the
template, and a test asserts the declared order matches the actual
placeholders **in all five languages** — the two cannot drift apart. The mock
builds the real request body for the same reason: a mock that skipped it would
never catch an ordering mistake.

One campaign per template per language, because WhatsApp approves templates per
language. No English fallback at the connector — that sends English under a
Tamil contact record.

The reminder and review request are queued with a send time, and the dispatcher
is the part that is easy to leave as a TODO. Three rules, all about *not*
sending:

| Situation | What happens | Why |
|---|---|---|
| Booking cancelled | message cancelled | Status is read at dispatch time. A patient who cancelled must not be told they're booked tomorrow. |
| Send window passed | **expired**, not sent | "Your appointment is tomorrow" arriving the day after is worse than silence. |
| Transport failure | stays queued | Retrying a 500 is free. |
| Template rejected | failed, no retry | Retrying is an infinite loop against an API that will never accept it. |

Expired is counted separately from cancelled because a rising expired count
means the dispatcher isn't running — an operational fault, not a patient
decision.

---

## What n8n is allowed to do

Glue: receive the delivery, forward it, fan out on the answer, run the
dispatcher on a schedule, retry, alert a human. That is routing, and routing is
what a visual workflow is good at.

What it must not do is **decide**. The rules that make this safe aren't
expressible as nodes — a threshold per slot, a read-back that clears a value
when rejected, an idempotency key evaluated under the same lock as the
availability check. A workflow can be edited by anyone with the editor open,
has no test suite, and fails in ways that produce a booking rather than an
error. The moment `if confidence > 0.9` appears in a Switch node, every
guarantee here is decorative.

So the contract is narrow: n8n calls `POST /webhooks/bolna` and reads
`outcome`. The workflow is **generated, not exported from the editor**, so it
can be diffed and asserted about — tests check that no node reaches a booking
endpoint, that no node parameter inspects a confidence value, that the Switch
keys match the service's own `Outcome` type (adding an outcome later breaks the
test rather than falling through a default), and that nothing key-shaped was
written into the committed JSON.

`integrations/n8n/clinic-receptionist.json` imports directly.

---

## The console

Served from the same process at `/console/` — plain HTML, CSS and JavaScript.
No npm, no bundler, no CDN. A clinic's own IT can read and edit it.

Four views: simulate a call and watch slots fill with live confidence; the
booking calendar; the WhatsApp queue, with a button that runs a dispatch pass
at any time you name, so the reminder-expires and cancelled-booking paths are
things you can watch rather than read about; and the accuracy table above.

Accessibility is part of the deliverable: skip link, semantic landmarks,
`scope`-ed table headers with captions, labelled controls with an error
summary, one polite live region for status, arrow-key tab navigation, status
carried by text and never by colour alone, AA contrast in both themes, and
`prefers-reduced-motion` honoured. It also ships the Noto font stack, without
which the Indic text the whole project is about renders as boxes.

---

## Architecture

```
Bolna ─webhook→ telephony/ingest ─┐
                                  ├→ workflow/call ⇄ nlu/slots → scheduling/calendar
console ────────────────────────*─┘         ↓
                                    messaging/dispatch → AiSensy
n8n: routes outcomes, runs the dispatcher on a schedule, alerts a human
```

Both entry points reach the calendar through the same state machine, which is
the whole point: there is one confidence gate, not one per channel.

| Layer | May do | May not do |
|---|---|---|
| `nlu` | Extract and score | Book anything |
| `scheduling` | Reserve atomically | Interpret speech |
| `workflow` | Sequence the call | Compute confidence |
| `telephony` | Parse and replay a payload | Extract or book directly |
| `messaging` | Render, send, schedule | Decide what to send |
| `api` | Transport | Contain logic |
| n8n | Route, schedule, retry, alert | Decide anything |

Connectors (Bolna, AiSensy) sit behind interfaces with mock implementations,
so the entire suite and demo run with no credentials. The AiSensy mock builds
the real request body, so an ordering mistake fails offline.

---

## What is deliberately not here

- **No real Bolna or AiSensy calls.** Both are integrated against their
  documented payload shapes and sit behind interfaces with mocks. I have no
  accounts, and pretending otherwise would make the tests meaningless. The
  n8n workflow is generated and valid for import, but has not been run against
  a live Bolna agent.
- **No ASR or TTS.** The system consumes transcripts. The evaluation harness
  feeds it word confidences to simulate degradation; the Bolna path has none,
  because Bolna does not send any — that difference is the design constraint
  described above, not an oversight. Degradation is simulated, never measured
  on real audio.
- **The Bolna agent prompt is not here.** Making the read-backs happen during
  the call is a prompt-engineering job on Bolna's side. This repository assumes
  they happened and verifies it from the transcript; it cannot make them happen.
- **The corpus is 11 utterances.** Enough to catch the bugs above; not enough
  to make a claim about production accuracy. Its thinness is not theoretical:
  every case happened to name a procedure outright, so a caller who described
  a symptom instead was met with the same question four times over. That
  transcript is now case `en-06`.
- **No auth, no persistence.** Sessions are in memory.
- **Slot extraction is rule-based.** For a closed schema with four fields,
  rules are inspectable, testable and free — and confidence means something
  specific rather than being a softmax the model reports about itself. The LLM
  is a *second* extractor bolted alongside, allowed only to lower confidence;
  it never produces a value that gets booked.
- **The cross-check is off by default and, measured on the corpus, currently
  buys nothing** — see the table above. It has never run against real call
  audio, and the corpus is ten utterances.
