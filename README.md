# Clinic Voice Receptionist

Multilingual inbound voice agent for dental clinics — English, Tamil, Kannada,
Malayalam and Hindi. Books appointments, sends WhatsApp follow-up, and
**never books on something it isn't sure it heard.**

```
$ python scripts/demo.py       # no API keys, no network, no telephony account
$ python -m pytest -q          # 69 passed
$ uvicorn receptionist.api.main:app --app-dir src   # console at localhost:8000
```

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
| 15% | 83.8% | 100.0% | 6 | **0** |
| 30% | 75.7% | 100.0% | 9 | **0** |
| 50% | 75.7% | 100.0% | 9 | **0** |

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

## The console

Served from the same process at `/console/` — plain HTML, CSS and JavaScript.
No npm, no bundler, no CDN. A clinic's own IT can read and edit it.

Four views: simulate a call and watch slots fill with live confidence; the
booking calendar; the WhatsApp queue with rendered message bodies per
language; and the accuracy table above.

Accessibility is part of the deliverable: skip link, semantic landmarks,
`scope`-ed table headers with captions, labelled controls with an error
summary, one polite live region for status, arrow-key tab navigation, status
carried by text and never by colour alone, AA contrast in both themes, and
`prefers-reduced-motion` honoured. It also ships the Noto font stack, without
which the Indic text the whole project is about renders as boxes.

---

## Architecture

```
telephony → nlu/language → nlu/slots ⇄ workflow/call → scheduling/calendar
                                            ↓
                                    messaging (WhatsApp)
```

| Layer | May do | May not do |
|---|---|---|
| `nlu` | Extract and score | Book anything |
| `scheduling` | Reserve atomically | Interpret speech |
| `workflow` | Sequence the call | Compute confidence |
| `api` | Transport | Contain logic |

Connectors (Bolna, AiSensy) sit behind interfaces with mock implementations,
so the entire suite and demo run with no credentials.

---

## What is deliberately not here

- **No real Bolna or AiSensy calls.** Both are behind interfaces with mocks.
  I have no accounts, and pretending otherwise would make the tests meaningless.
- **No ASR or TTS.** The system consumes transcripts with word confidences —
  which is what Bolna's webhook delivers. Degradation is simulated, not measured
  on real audio.
- **The corpus is 10 utterances.** Enough to catch the bugs above; not enough
  to make a claim about production accuracy.
- **No auth, no persistence.** Sessions are in memory.
- **Slot extraction is rule-based, not an LLM.** For a closed schema with four
  fields, rules are inspectable, testable and free — and confidence means
  something specific rather than being a softmax the model reports about itself.
