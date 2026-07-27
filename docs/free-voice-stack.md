# A voice loop that costs nothing to run

Target: a caller talks to the agent and hears it answer, with no metered
service anywhere in the path. Free of charge is the requirement — a hosted
service is fine if it does not bill.

## What has to go, and what it costs to lose it

| Component | Today | Free replacement | What is lost |
|---|---|---|---|
| Telephony | Bolna number (~$5/mo + per-minute) | **Browser WebRTC** | The number. Nobody can dial the clinic. |
| Speech-to-text | Bolna's built-in | **faster-whisper** / **IndicConformer** | Nothing — arguably a gain, see below |
| Text-to-speech | Bolna's built-in | **IndicF5** / **Indic Parler-TTS** | Voice quality, and latency on CPU |
| Turn-taking, barge-in | Bolna | **LiveKit Agents** (Apache-2.0) | Has to be assembled rather than configured |
| Notifications | AiSensy + WhatsApp | **Telegram** *(done)* | WhatsApp itself — see below |
| Workflow glue | n8n | n8n community *(already self-hosted)* | Nothing |
| LLM cross-check | `gpt-oss:120b-cloud` | *unchanged* — Ollama Cloud does not bill | Nothing |

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

```
browser mic ──WebRTC──▶ LiveKit ──▶ VAD ──▶ ASR ──▶ /calls/{id}/utterance
                            ▲                              │
                            └────── TTS ◀── agent reply ◀───┘
```

The service in this repository is unchanged. It already exposes exactly the
two things a voice loop needs — `POST /calls` and
`POST /calls/{id}/utterance` — and the console has been driving them as text
since the beginning. The voice work is entirely about getting audio in and
out of those endpoints.

### Components

- **Transport — LiveKit** (Apache-2.0, self-hosted). Handles WebRTC, jitter,
  echo cancellation and multi-participant rooms. The alternative is raw
  `RTCPeerConnection` plus a signalling server, which is less to install and
  considerably more to get right.
- **Endpointing — Silero VAD** (MIT). Deciding *when the caller has stopped
  talking* is the difference between a conversation and a walkie-talkie, and
  it is the part most likely to feel broken first.
- **ASR — faster-whisper or IndicConformer.** Both free, both local. The
  choice is argued in `asr/base.py`: Whisper has the better WER and fabricates
  fluent text at high confidence on silence, which is adversarial to the
  confidence gate; IndicConformer's CTC branch gives real per-word posteriors.
- **TTS — IndicF5** (AI4Bharat, 11 Indian languages including all four
  non-English ones here) or **Indic Parler-TTS**. Both self-hostable and free.

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

1. **Text loop end to end over WebRTC** — no ASR, no TTS. Prove transport,
   endpointing and session plumbing while failures are still legible.
2. **Add ASR.** Now the confidence gate is being fed real numbers for the
   first time; expect more read-backs, not fewer.
3. **Add TTS.** The first point at which the thing can be heard.
4. **Measure.** Replace the harness's simulated word corruption with real
   audio at 8 kHz. That retires the weakest claim in the README.

Step 4 is the one that matters for the project's own thesis, and it is the
one most likely to be skipped.
