# A voice loop that costs nothing to run

**Status: built.** A caller talks to the agent in the browser and hears it
answer, with no metered service in the path. What follows is what was chosen
and why, and what is still missing.

Measured on an RTX 4050 Laptop: Whisper `small` transcribes 6.9 s of audio in
0.28 s, Piper synthesises at RTF 0.05, MMS covers Tamil and Kannada at
0.17–0.20 s, and a full turn — audio in, gate, spoken reply — lands at
0.8–1.0 s after a 7 s cold start.

The two things that remain true regardless: **a dialable phone number cannot
be free**, so the caller reaches the clinic through a web page rather than a
handset; and **MMS-TTS is CC-BY-NC**, fine for a portfolio piece and not for a
clinic that charges patients.

## What has to go, and what it costs to lose it

| Component | Today | Free replacement | What is lost |
|---|---|---|---|
| Telephony | Bolna number (~$5/mo + per-minute) | **Browser WebRTC** | The number. Nobody can dial the clinic. |
| Speech-to-text | Bolna's built-in | **faster-whisper** *(done)* | Whisper's confidence is weaker than a CTC posterior — see below |
| Text-to-speech | Bolna's built-in | **Piper** + **MMS** *(done)* | MMS is non-commercial; Piper has no Tamil or Kannada |
| Turn-taking, barge-in | Bolna | push-to-talk *(done)*, LiveKit later | No barge-in, and the caller holds a button |
| Notifications | AiSensy + WhatsApp | **Telegram** *(done)* | WhatsApp itself — see below |
| Workflow glue | n8n | n8n community *(already self-hosted)* | Nothing |
| Understanding | rules | **`gpt-oss:120b-cloud`** *(done)* | Free of charge, but transcripts leave the machine — opt-in |

### The two that cannot be fixed by engineering

**A dialable number is a metered, regulated resource.** There is no free
PSTN DID, and services that appear to offer one are trials. Free means the
caller reaches the clinic through a web page, not a phone. For a portfolio
piece that is arguably *better* — a reviewer can click and talk immediately
instead of dialling a number that costs you money per minute — but it is not
the product a clinic asked for.

**WhatsApp is Meta's, and Meta bills per template message** from the first
send; the free-conversation tier is gone as of 2026. Swapping AiSensy for
another BSP removes the BSP's cut, not Meta's. Telegram has no per-message
cost, which is why the connector now exists — but Telegram addresses a
`chat_id`, not a phone number, so every patient has to press Start before the
clinic can message them at all. That is a real operational cost paid in
enrolment rather than in money.

## The shape of the free voice path

What was built, which is simpler than the plan:

```
browser mic ──MediaRecorder──▶ POST /calls/{id}/audio
                                      │
              Whisper ──▶ LLM understanding ──▶ confidence gate
                                      │
              browser ◀── WAV ◀── Piper / MMS ◀── agent reply
```

`POST /calls` and `POST /calls/{id}/utterance` already existed and the console
had been driving them as text from the start, so the voice work was entirely
about getting audio in and out of those endpoints.

### Components, as chosen

- **Transport — push-to-talk, not WebRTC.** Hold the button, speak, release.
  Endpointing — deciding when a caller has *stopped* — is its own hard
  problem, and doing it badly makes a system feel broken for reasons
  unrelated to whether the booking logic is right. The button sidesteps it
  honestly rather than half-solving it. LiveKit and Silero VAD are the path
  to a real conversation, and neither is here.
- **ASR — faster-whisper**, against the argument in `asr/base.py`, which
  makes the case for a CTC model because Whisper fabricates fluent text at
  high confidence on silence. IndicConformer needs NeMo, NeMo needs torch,
  and every AI4Bharat repository is gated. Whisper is what runs, so its
  failure mode is mitigated instead: Silero VAD ahead of the decoder, and
  `no_speech_prob` discounting every word from a doubted segment.
- **TTS — Piper for English, Hindi and Malayalam; MMS for Tamil and
  Kannada.** Piper has no voice for those two at all. IndicF5 was tried first
  and is unusable here: it pins `numpy<=1.26.4` against a Python that needs
  numpy 2, and installing it broke torch outright.

### Latency is the whole engineering problem

A phone conversation tolerates roughly 300–500 ms of silence before it feels
broken. The budget:

| Stage | CPU | GPU |
|---|---|---|
| VAD endpointing | ~50 ms | ~50 ms |
| ASR on a 5 s utterance | 1–3 s | 150–400 ms |
| Slot extraction (this repo) | <5 ms | <5 ms |
| TTS for a 2 s reply | 1–2 s | 100–300 ms |

**On CPU this does not work.** Two to five seconds per turn is not a
conversation. A consumer GPU brings it inside the budget; without one, the
honest options are streaming ASR with partial results, a shorter reply, or
accepting that the demo is visibly slow.

## Why the existing design absorbs this well

Three interfaces already exist for exactly these swaps, and each was written
before there was a second implementation to justify it:

- `MessagingConnector` — AiSensy and Telegram now sit behind it, and the call
  flow cannot tell which is in use. `test_the_whole_call_flow_works_on_telegram`
  is that claim being checked rather than stated.
- `Transcriber` — audio in, words with confidence out. The mock keeps the
  suite offline; a real model drops in behind it.
- `/calls/{id}/utterance` — text in, agent reply out. Whatever produces the
  text, the gate is the same one.

The confidence model gains something real from this move. Bolna sends a plain
transcript with no per-word confidence, so on that path every slot falls back
to a default. A local ASR run produces genuine posteriors, which is the input
`_asr_confidence` was designed around and has never actually received.

## Order of work

1. ~~Text loop end to end~~ — **done**, as push-to-talk rather than WebRTC.
2. ~~Add ASR~~ — **done**. The confidence gate is fed real per-word numbers
   for the first time on the voice path.
3. ~~Add TTS~~ — **done**, all five languages, Piper where it has a voice and
   MMS for the two it does not.
4. **Measure.** Still outstanding, and still the one that matters. Every
   accuracy figure in the README comes from simulated word corruption; none
   comes from sound. Replacing that with real 8 kHz telephony audio would
   retire the weakest claim in the project.
5. **Real endpointing.** Push-to-talk sidesteps the hard problem honestly, but
   a phone caller cannot hold a button. Silero VAD plus LiveKit is the path.

Step 4 was predicted to be the one most likely to be skipped, and it has
been.
