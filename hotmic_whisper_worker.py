#!/usr/bin/env python3
"""
Persistent whisper daemon for hotmic.

Loads the model ONCE, then loops accepting audio sessions via a FIFO:
1. Opens FIFO for reading (blocks until sox connects)
2. Reads PCM audio, splits on silence or max duration
3. Transcribes each chunk and types into the saved target window
4. When sox dies (EOF on FIFO), transcribes remaining audio
5. Goes back to step 1 — model stays loaded in GPU memory

This eliminates the ~26s model reload penalty on every dictation session.
"""

import os
import sys
import wave
import struct
import signal
import math
import time
import queue
import subprocess
import traceback
from collections import deque
from threading import Thread, Event, Lock

DIR = os.environ.get("HOTMIC_DIR", "/tmp/hotmic")  # overridable for isolated tests
# Created by hotmic_start.sh before it launches sox, removed by hotmic_stop.sh —
# present for the whole dictation lifecycle. The watchdog uses it to avoid
# re-exec'ing the daemon while a session is starting (sox connected but audio not
# yet flowing — a state the FIFO itself can't distinguish from idle).
STATE_FILE = f"{DIR}/active"
READY_FILE = f"{DIR}/whisper.ready"
LOG_FILE = f"{DIR}/hotmic.log"
CHUNK_DIR = f"{DIR}/chunks"
FIFO_PATH = f"{DIR}/audio.fifo"
WINDOW_FILE = f"{DIR}/window_id"

# Audio format (must match sox output)
RATE = 16000
CHANNELS = 1
SAMPWIDTH = 2  # 16-bit

# Splitting config (overridable via env)
MAX_CHUNK_SEC = int(os.environ.get("MAX_CHUNK_SEC", "20"))
SILENCE_DUR = float(os.environ.get("SILENCE_DUR", "0.8"))
SILENCE_THRESH = float(os.environ.get("SILENCE_THRESH", "0.03"))  # 3% as fraction

# Analysis block size
BLOCK_SAMPLES = int(RATE * 0.05)  # 50ms blocks
BLOCK_BYTES = BLOCK_SAMPLES * SAMPWIDTH
SILENCE_BLOCKS = int(SILENCE_DUR / 0.05)  # blocks of silence needed to split
MIN_CHUNK_SAMPLES = int(RATE * 0.3)  # ignore chunks shorter than 0.3s

# Continuous capture: the daemon holds the mic open and fills a rolling ring
# buffer, so a dictation is a [t_start - LOOKBACK, t_stop] window over an
# always-live stream — no startup gap, and we keep audio from just before the
# keypress. See docs/superpowers/specs/2026-06-04-continuous-capture-ring-buffer.
RING_SECONDS = float(os.environ.get("RING_SECONDS", "10"))
LOOKBACK_SEC = float(os.environ.get("LOOKBACK_SEC", "2.0"))
RING_BLOCKS = max(1, int(RING_SECONDS / 0.05))  # 50ms blocks -> 200 for 10s
CONTROL_FIFO = f"{DIR}/control.fifo"
PAUSED_FLAG = f"{DIR}/paused"
# Persistent mic capture (raw 16k mono s16le to stdout). HOTMIC_SOURCE pins an
# explicit ALSA device; default uses sox's default input (-d).
CAPTURE_CMD_DEFAULT = ["sox", "-q", "-d", "-t", "raw",
                       "-r", str(RATE), "-c", str(CHANNELS), "-b", "16",
                       "-e", "signed-integer", "-"]

# Idle restart: after this many seconds with no dictation, re-exec the daemon.
# A full process restart is the ONLY thing that reclaims the multi-GB host RAM
# CTranslate2/CUDA pool up over transcriptions (~150 MB each) — model unload and
# malloc_trim do NOT return it. It also frees the GPU and clears accumulated
# X/xdotool state, and re-runs main() fresh while keeping the daemon resident +
# ready. main() now pre-warms the model in the background immediately after the
# re-exec, so the next dictation hits the warm path with no reload latency — the
# 20 min window is just the RAM-reset cadence, no longer a warm-vs-cold gate.
# Steady-state RSS with the model resident is ~1.4 GB, which is acceptable.
RESTART_IDLE_SEC = int(os.environ.get("RESTART_IDLE_SEC", "1200"))  # default 20 min
# How often the watchdog re-checks idle time (lower only for tests).
WATCHDOG_INTERVAL_SEC = int(os.environ.get("WATCHDOG_INTERVAL_SEC", "30"))


def log(msg):
    from datetime import datetime
    with open(LOG_FILE, "a") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] whisper-daemon: {msg}\n")


def rms(samples):
    """Calculate RMS of 16-bit PCM samples."""
    if not samples:
        return 0.0
    sum_sq = sum(s * s for s in samples)
    return math.sqrt(sum_sq / len(samples)) / 32768.0


def samples_to_wav(raw_samples, path):
    """Write raw 16-bit PCM samples to a WAV file."""
    data = struct.pack(f"<{len(raw_samples)}h", *raw_samples)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPWIDTH)
        wf.setframerate(RATE)
        wf.writeframes(data)


def get_target_window():
    """Read the saved window ID to type into."""
    try:
        with open(WINDOW_FILE) as f:
            wid = f.read().strip()
            return wid if wid else None
    except OSError:
        return None


class RingBuffer:
    """Rolling buffer of (monotonic_ts, raw_bytes) blocks, plus an optional
    session tee. The lock makes the lookback snapshot and the tee-arm atomic, so
    there is no gap or duplicate block at the session-start seam."""

    def __init__(self, maxlen):
        self._dq = deque(maxlen=maxlen)
        self._lock = Lock()
        self._tee = None   # queue.Queue while a session is active

    def append(self, ts, block):
        with self._lock:
            self._dq.append((ts, block))
            if self._tee is not None:
                self._tee.put(block)

    def snapshot(self):
        with self._lock:
            return list(self._dq)

    def start_session(self, t_start, lookback_sec):
        """Seed a fresh session queue with the lookback blocks and arm the tee,
        atomically. Returns the queue."""
        q = queue.Queue()
        with self._lock:
            for ts, block in self._dq:
                if ts >= t_start - lookback_sec:
                    q.put(block)
            self._tee = q
        return q

    def stop_session(self):
        """Disarm the tee and push the end sentinel (None) onto the queue."""
        with self._lock:
            q = self._tee
            self._tee = None
        if q is not None:
            q.put(None)
        return q


def reader_loop(audio_input, chunk_queue, session_stop):
    """Read continuous PCM from audio_input and split into chunks."""
    chunk_num = 0
    audio_buf = []
    silent_blocks = 0
    has_speech = False
    max_samples = MAX_CHUNK_SEC * RATE

    try:
        while not session_stop.is_set():
            raw = audio_input.read(BLOCK_BYTES)
            if not raw:
                break  # EOF — sox closed the FIFO

            # Decode PCM samples
            n_samples = len(raw) // SAMPWIDTH
            samples = struct.unpack(f"<{n_samples}h", raw[:n_samples * SAMPWIDTH])
            audio_buf.extend(samples)

            # Analyse block
            block_rms = rms(samples)
            is_silent = block_rms < SILENCE_THRESH

            if not is_silent:
                has_speech = True
                silent_blocks = 0
            else:
                silent_blocks += 1

            # Split conditions
            should_split = False
            if has_speech and silent_blocks >= SILENCE_BLOCKS and len(audio_buf) >= MIN_CHUNK_SAMPLES:
                should_split = True  # natural pause
            elif len(audio_buf) >= max_samples:
                should_split = True  # hard cap

            if should_split:
                chunk_path = os.path.join(CHUNK_DIR, f"chunk_{chunk_num}.wav")
                samples_to_wav(audio_buf, chunk_path)
                log(f"chunk {chunk_num}: {len(audio_buf)} samples ({len(audio_buf)/RATE:.1f}s) → queue")
                chunk_queue.append((chunk_path, chunk_num))
                chunk_num += 1
                audio_buf = []
                silent_blocks = 0
                has_speech = False

    except Exception as e:
        log(f"reader error: {e}")

    # Flush remaining audio — always flush at session end if we have enough
    # samples. The user explicitly stopped, so transcribe whatever's left.
    # (Whisper returns empty text for silence, which we skip anyway.)
    if audio_buf and len(audio_buf) >= MIN_CHUNK_SAMPLES:
        chunk_path = os.path.join(CHUNK_DIR, f"chunk_{chunk_num}.wav")
        samples_to_wav(audio_buf, chunk_path)
        log(f"final chunk {chunk_num}: {len(audio_buf)} samples ({len(audio_buf)/RATE:.1f}s) → queue")
        chunk_queue.append((chunk_path, chunk_num))

    log("reader finished")


def transcriber_loop(model_holder, chunk_queue, session_stop, window_id):
    """Transcribe chunks and type results into the target window."""
    # Wait for model if still loading (first session may start before model is ready)
    deadline = time.monotonic() + 120
    while not model_holder:
        if session_stop.is_set() and not chunk_queue:
            return  # stopped with nothing to transcribe
        if time.monotonic() > deadline:
            log("model load timeout, giving up on session")
            return
        time.sleep(0.1)

    model = model_holder[0]

    while not session_stop.is_set() or chunk_queue:
        if not chunk_queue:
            session_stop.wait(0.05)
            continue

        chunk_path, chunk_num = chunk_queue.popleft()

        if not os.path.isfile(chunk_path):
            continue

        size = os.path.getsize(chunk_path)
        log(f"transcribing chunk {chunk_num} ({size} bytes)")

        try:
            segments, _ = model.transcribe(
                chunk_path,
                language="en",
                beam_size=1,
                temperature=0,
            )
            text = " ".join(s.text for s in segments).strip()
        except Exception as e:
            log(f"transcribe error: {e}")
            text = ""
        finally:
            try:
                os.remove(chunk_path)
            except OSError:
                pass

        # Skip empty or non-speech transcriptions. Whisper emits punctuation-only
        # noise ("." / ".." / "..." / ". .") for silence and breaths; typing that
        # is worse than nothing. Requiring at least one alphanumeric character
        # drops the noise without risking real short words ("you", "ok", "no").
        if not any(c.isalnum() for c in text):
            log(f"skipping non-speech transcription: {text!r}")
            continue

        log(f"transcribed: {text}")
        try:
            if window_id:
                # Briefly focus the target window, type, then let the user's
                # current window naturally retain focus. windowactivate is more
                # reliable than --window (which sends synthetic events many
                # apps ignore).
                subprocess.run(
                    ["xdotool", "windowactivate", "--sync", window_id],
                    timeout=2,
                )
            subprocess.run(
                ["xdotool", "type", "--clearmodifiers", "--delay", "0", "--", text + " "],
                timeout=5,
            )
        except Exception as e:
            log(f"xdotool error: {e}")


def main():
    model_name = os.environ.get("WHISPER_MODEL", "medium.en")
    device = os.environ.get("WHISPER_DEVICE", "cuda")
    compute_type = "int8" if device == "cuda" else "float32"

    os.makedirs(CHUNK_DIR, exist_ok=True)

    # Create FIFO
    if os.path.exists(FIFO_PATH) and not stat_is_fifo(FIFO_PATH):
        os.remove(FIFO_PATH)
    if not os.path.exists(FIFO_PATH):
        os.mkfifo(FIFO_PATH)

    # Graceful shutdown
    daemon_stop = Event()
    def handle_signal(signum, frame):
        daemon_stop.set()
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Model holder — shared between sessions, populated by loader thread.
    # Using a list so threads can see when it's loaded/unloaded.
    model_holder = []
    model_loading = Event()  # set while a load is in progress

    def load_model():
        """Load whisper model into GPU (or CPU fallback)."""
        if model_holder or model_loading.is_set():
            return
        model_loading.set()
        try:
            log(f"loading model={model_name} device={device} compute_type={compute_type}")
            from faster_whisper import WhisperModel
            m = WhisperModel(model_name, device=device, compute_type=compute_type)
            model_holder.append(m)
            log("model loaded, ready for chunks")
        except Exception as e:
            if device != "cpu":
                log(f"CUDA failed ({e}), falling back to CPU")
                try:
                    m = WhisperModel(model_name, device="cpu", compute_type="float32")
                    model_holder.append(m)
                    log("model loaded on CPU, ready for chunks")
                except Exception as e2:
                    log(f"FATAL: failed to load model: {e2}")
                    traceback.print_exc(file=open(LOG_FILE, "a"))
            else:
                log(f"FATAL: failed to load model: {e}")
                traceback.print_exc(file=open(LOG_FILE, "a"))
        finally:
            model_loading.clear()

    # Track last session time for idle timeout
    last_session_end = [time.monotonic()]
    # Set while a dictation session is active, so the watchdog never re-execs the
    # process underneath a running session.
    in_session = Event()

    # A dictation is active (or starting) if either the in_session flag is set OR
    # start.sh's STATE_FILE exists. The STATE_FILE is the authoritative signal: it
    # is created before sox launches and removed on stop, so it covers the window
    # where sox has connected but in_session isn't set yet — a state the FIFO
    # cannot distinguish from idle.
    def dictation_active():
        return in_session.is_set() or os.path.exists(STATE_FILE)

    # Watchdog thread: re-execs the whole process after RESTART_IDLE_SEC of
    # inactivity. A full restart is the ONLY way to reclaim the multi-GB host RAM
    # CTranslate2/CUDA pool up over transcriptions (model unload / malloc_trim
    # don't return it); it also frees the GPU and clears accumulated X/xdotool
    # state. It replaces the old SIGTERM (which the main thread — blocked in
    # open(FIFO) — never observed, so the process just lingered and the NEXT
    # keypress paid a cold-spawn gap that ate the start of dictation) and re-runs
    # main() fresh while keeping the daemon resident + ready. main() pre-warms the
    # model in the background right after re-exec, so a recording after the idle
    # restart still hits the warm path (no reload delay) unless it lands in the
    # few seconds before the reload finishes. It never fires while a
    # dictation is active or starting (dictation_active guard, re-checked just
    # before execv).
    def idle_watchdog():
        while not daemon_stop.is_set():
            daemon_stop.wait(WATCHDOG_INTERVAL_SEC)
            if daemon_stop.is_set():
                return
            if dictation_active():
                continue  # never restart during/just before a session
            idle = time.monotonic() - last_session_end[0]
            if idle > RESTART_IDLE_SEC:
                # Re-check immediately before execv to close the tiny window
                # against a session starting since the guard above.
                if dictation_active():
                    continue
                log(f"idle for {int(idle)}s, re-exec for fresh process "
                    f"(frees GPU + reclaims host RAM; daemon stays resident + ready)")
                # Keep whisper.ready and the FIFO in place so start.sh keeps seeing
                # a ready resident daemon (warm path). execv replaces this entire
                # process image — including the main thread blocked in open(FIFO) —
                # and re-runs main() fresh, inheriting the current environment
                # (LD_LIBRARY_PATH, WHISPER_*, MAX_CHUNK_SEC, ...).
                os.execv(sys.executable, [sys.executable, os.path.abspath(__file__)])
                return  # unreachable if execv succeeds

    watchdog = Thread(target=idle_watchdog, daemon=True)
    watchdog.start()

    # Pre-warm the model eagerly in the background so the daemon is ready to
    # transcribe the instant the next dictation lands — no first-session reload
    # delay. This runs on every fresh start AND after every idle re-exec (execv
    # re-runs main()), so a recording after the idle restart hits the warm path
    # too. Loading in a background thread keeps readiness immediate; the reader
    # buffers audio and the transcriber waits if a session starts mid-load. The
    # lazy load path in the session loop below remains a fallback (e.g. if this
    # pre-warm failed). Holding the model resident is ~1.4 GB RSS — acceptable now
    # that runaway growth is bounded.
    Thread(target=load_model, daemon=True).start()

    # Signal readiness — FIFO exists, daemon is accepting sessions.
    open(READY_FILE, "w").close()
    log(f"daemon ready, pre-warming model (idle restart: {RESTART_IDLE_SEC}s)")

    # === Main session loop ===
    while not daemon_stop.is_set():
        # Open FIFO for reading — blocks until sox opens it for writing. While
        # blocked here the daemon is idle, so the watchdog may re-exec the process.
        try:
            fifo = open(FIFO_PATH, "rb")
        except OSError as e:
            if daemon_stop.is_set():
                break
            log(f"FIFO open error: {e}")
            time.sleep(0.5)
            continue

        # A writer (sox) connected — mark the session active so the watchdog won't
        # unload the model under us. The window between this open() returning and
        # here was already guarded by start.sh's STATE_FILE.
        in_session.set()
        try:
            # Load the model if it isn't resident — the lazy first-load path and
            # the reload-after-idle-unload path. The reader buffers audio while
            # this runs in the background; the transcriber waits for it.
            if not model_holder and not model_loading.is_set():
                log("model not loaded, loading for new session...")
                loader = Thread(target=load_model, daemon=True)
                loader.start()

            window_id = get_target_window()
            log(f"session started (target window: {window_id or 'active'})")

            chunk_queue = deque()
            session_stop = Event()

            reader = Thread(target=reader_loop, args=(fifo, chunk_queue, session_stop))
            reader.start()

            transcriber = Thread(target=transcriber_loop, args=(model_holder, chunk_queue, session_stop, window_id))
            transcriber.start()

            # Wait for reader to finish (EOF = sox died / was killed)
            reader.join()

            # Wait for transcriber to drain remaining chunks
            drain_deadline = time.monotonic() + 30
            while chunk_queue and time.monotonic() < drain_deadline:
                time.sleep(0.1)

            session_stop.set()
            transcriber.join(timeout=15)

            try:
                fifo.close()
            except OSError:
                pass

            last_session_end[0] = time.monotonic()
            log("session complete")
        finally:
            in_session.clear()

    # Cleanup
    try:
        os.remove(READY_FILE)
    except OSError:
        pass
    try:
        os.remove(FIFO_PATH)
    except OSError:
        pass
    log("daemon exiting")


def stat_is_fifo(path):
    """Check if path is a FIFO."""
    import stat
    try:
        return stat.S_ISFIFO(os.stat(path).st_mode)
    except OSError:
        return False


if __name__ == "__main__":
    main()
