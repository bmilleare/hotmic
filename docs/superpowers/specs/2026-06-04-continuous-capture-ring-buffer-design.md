# Continuous-Capture Ring Buffer — Design Spec

**Date:** 2026-06-04
**Status:** Approved (pending spec review)
**Component:** `hotmic_whisper_worker.py`, `hotmic_start.sh`, `hotmic_stop.sh`, new `hotmic_pause.sh`

## Problem

The first dictation after a long idle (e.g. returning from lunch) loses the
opening ~2–4 s of speech. The transcription begins mid-sentence. Investigation
(see RCA in conversation 2026-06-04) showed the loss is **upstream of whisper**:
the audio-capture path is not live at the instant the user starts speaking after
a long idle. The whisper daemon faithfully transcribes whatever `sox` delivers;
the opening words never reach it.

Candidate causes (mic suspend/resume, paged-out launch toolchain delaying `sox`,
idle memory pressure from the model pre-warm) all share one shape: a **cold-start
gap** between the keypress and `sox` actually capturing valid audio. Rather than
pin down which contributor dominates, we remove the startup gap entirely so the
bug is **structurally impossible**.

## Goal

Capture must already be live when the user starts speaking, and must include a
short window of audio from *before* the keypress. No startup gap, ever (except
immediately after an explicit pause→resume, which is an accepted trade-off).

## Approach: daemon owns continuous capture; sessions are a window over the stream

The daemon — already resident — takes ownership of the microphone and records it
continuously into a small in-RAM ring buffer. A dictation session becomes a
marked `[t_start − LOOKBACK, t_stop]` window over that always-running stream.
`start.sh` no longer spawns `sox`.

### Architecture

```
                 persistent sox (mic → raw PCM, 16kHz mono 16-bit)
                              │
                   ┌──────────▼───────────┐
                   │   capture thread     │  always running while ARMED
                   │  read 50ms blocks    │
                   └──────┬─────────┬──────┘
                          │         │ (only while a session is active)
              ring buffer │         │ tee
              (deque of   │         ▼
              raw bytes,  │   session block queue  ← preloaded with LOOKBACK
              last ~10s)  │         │                on START
                          │         ▼
                          │     chunker  (RMS silence-split / MAX_CHUNK)
                          │         │  writes chunk WAVs
                          │         ▼
                          │     chunk_queue
                          │         │
                          │         ▼
                          │   transcriber  (whisper → xdotool type)
                          │
        control FIFO ─────┴──▶ control thread  (START / STOP / PAUSE / RESUME)
```

### Components

**Capture thread (always running while armed).** Reads raw PCM from the
persistent `sox` subprocess in fixed 50 ms blocks (`BLOCK_BYTES` = 1600 bytes).
Under the shared lock, it appends each block to the ring buffer as **raw bytes**
(no decode, no RMS) and, if a session is active, also enqueues the same block onto
the session block queue — both in one locked critical section. This loop does
*zero* signal processing when idle — store + evict only (measured at 0.0003 % of
one core).

**Ring buffer.** A `deque(maxlen=RING_BLOCKS)` of `(monotonic_ts, raw_bytes)`
tuples, sized to `RING_SECONDS` (default 10 s → 200 blocks ≈ 312 KB). Holds only
the rolling tail; its sole job is to supply the LOOKBACK preload on START. Access
guarded by a lock (capture appends; control thread snapshots on START).

**Control thread.** Opens the control FIFO `O_RDWR` (so it never sees EOF and
external writers never block) and reads newline-delimited commands:
`START`, `STOP`, `PAUSE`, `RESUME`. Dispatches to session/pause logic.

**Session start (START).** Record `t_start = monotonic()`. **In one locked
critical section** (the same lock the capture thread uses): snapshot from the ring
buffer every block with `ts >= t_start − LOOKBACK_SEC` (default **0.5 s**) to seed
the session block queue, then set the session active so the capture thread begins
teeing subsequent blocks. Holding the lock across both steps makes the seam clean
— every block already in the ring is in the snapshot, every block appended after
is teed, so there is no gap or duplicate at the boundary. Then spawn the chunker +
transcriber threads and read `window_id` from its file (unchanged mechanism).

**Chunker (active during session).** Consumes the session block queue, unpacks
each block, computes RMS, and applies the **existing** silence/`MAX_CHUNK_SEC`
splitting logic to emit chunk WAVs onto `chunk_queue`. This is the only place RMS
runs (measured 0.088 % of one core, sessions only). It is essentially today's
`reader_loop`, reading from a queue instead of the FIFO and pre-seeded with the
lookback blocks.

**Transcriber (active during session).** Unchanged from today: pops chunk WAVs,
transcribes with the resident model, skips non-speech, types into `window_id`
via `xdotool`.

**Session stop (STOP).** Capture thread stops teeing; a sentinel is pushed onto
the session block queue; the chunker flushes any remaining audio (`>= MIN_CHUNK`)
and exits; the transcriber drains `chunk_queue` and finalizes. `last_session_end`
is updated (feeds the idle watchdog).

**Pause / resume.** `PAUSE` kills the persistent `sox` (mic released, LED off),
creates a `$DIR/paused` flag, and stops the capture thread. `RESUME` removes the
flag and respawns `sox` + capture thread. The first dictation after a resume has
no lookback available and pays the normal cold-start once — the documented
trade-off. The `paused` flag is checked by `main()` on startup so a re-exec while
paused comes back paused (does not silently re-acquire the mic).

### Existing machinery — kept

- **Model pre-warm** (eager load at daemon startup): kept. First transcription is
  still fast.
- **45-min idle re-exec** (`RESTART_IDLE_SEC`, default `2700`): kept for RAM
  hygiene. "Idle" still means *no active session*. The re-exec path must **kill the
  persistent `sox` child before `os.execv`** (otherwise it is orphaned writing to a
  dead pipe); the new `main()` respawns capture. The brief capture gap happens only
  while idle, so no words are lost. Never fires mid-session (existing
  `dictation_active()` guard).
- **`window_id`, indicator, xdotool typing, non-speech skipping**: unchanged.
- **`STATE_FILE` (`$DIR/active`)**: kept as the watchdog "session active/starting"
  guard and as `start.sh`'s authoritative signal.

### Control-plane changes to scripts

- **`hotmic_start.sh`**: keep "ensure daemon running + ready" block; keep
  `xdotool getactivewindow > window_id`; keep `touch active` and indicator launch.
  **Replace** the `sox → FIFO` launch with writing `START` to the control FIFO.
- **`hotmic_stop.sh`**: **replace** sox/FIFO teardown with writing `STOP` to the
  control FIFO; keep removing `active` and killing the indicator. `--daemon` still
  fully stops the daemon.
- **`hotmic_pause.sh`** (new): toggles `PAUSE`/`RESUME` over the control FIFO based
  on the `$DIR/paused` flag. Intended to be bound to a hotkey by the user.

### Configuration (env, with defaults)

| Var | Default | Meaning |
|---|---|---|
| `RING_SECONDS` | `10` | Ring-buffer span (lookback + slack) |
| `LOOKBACK_SEC` | `0.5` | Audio retained from before the keypress (was 2.0; trimmed — continuous capture means no startup gap to cover, so a short cushion avoids grabbing unwanted prior words) |
| `HOTMIC_SOURCE` | `` (sox `-d`) | Optional explicit capture device |
| `RESTART_IDLE_SEC` | `2700` | Idle re-exec (raised 20→45 min) |
| `MAX_CHUNK_SEC`, `SILENCE_DUR`, `SILENCE_THRESH` | as today | Chunker splitting |

## Resource cost (measured 2026-06-04)

Armed-idle (the new continuous cost): **~1 % of one CPU core** (essentially all
`sox`; Python idle loop 0.0003 %), **~10 MB RAM** (`sox` ~9 MB + 312 KB buffer),
**zero GPU compute**, no disk. During transcription: **unchanged** from today.
Honest caveat: idle goes from ~0 % (today's daemon sleeps in `open(FIFO)`) to
~1 % of one core, because the process now wakes ~20×/s to drain `sox`. That tiny
constant cost is the mechanism — we keep the pipeline permanently warm.

Privacy: audio lives only in a RAM ring buffer, overwritten every ~10 s, never
written to disk unless a session is active. The mic LED is on whenever armed;
`hotmic_pause.sh` fully releases the device.

## Error handling / edge cases

- **START while paused**: auto-resume (respawn `sox`), then proceed cold (no
  lookback for that first session). Logged.
- **Duplicate START** (already in session): ignored.
- **STOP with no active session**: no-op.
- **`sox` dies unexpectedly** (e.g. device unplugged): capture thread detects
  EOF/error and respawns with backoff; logs. If a session is active, the chunker
  finalizes whatever audio is buffered.
- **Daemon not running when a script writes to the control FIFO**: `start.sh`
  ensures the daemon is up + ready first (existing logic, adapted — it no longer
  needs `sox`, only a ready daemon). Writes use a short timeout as a backstop.
- **Re-exec while paused**: `main()` sees `$DIR/paused` and comes up without
  capture.

## Testing

- **Unit — lookback selection**: feed synthetic `(ts, block)` sequence into the
  ring buffer; assert the START snapshot returns exactly the blocks within
  `[t_start − LOOKBACK, t_start]`.
- **Unit — chunker**: feed a crafted block sequence (speech/silence pattern);
  assert chunk boundaries match the silence/`MAX_CHUNK` rules (covers the logic
  lifted from `reader_loop`).
- **Integration — capture→type**: override `HOTMIC_DIR`; drive the capture source
  from a fixed WAV (instead of the live mic); inject a **fake transcriber** (mock
  `WhisperModel.transcribe`) for speed/determinism; send `START`/`STOP` over the
  control FIFO; assert the expected text is "typed" (mock `xdotool`) and that the
  output includes the lookback region.

## Deployment

All four files change and must be redeployed to `~/bin/` (live daemon runs from
there; deploy = manual `cp` + daemon restart). The control FIFO and `paused` flag
live under `$DIR` (`/tmp/hotmic`). The user should bind `hotmic_pause.sh` to a
hotkey.

## Out of scope

- A distinct indicator state for "paused" (could be added later).
- Recovering audio from before an explicit pause (impossible by design — mic is
  released while paused).
