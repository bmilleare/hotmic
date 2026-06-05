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
LOOKBACK_SEC = float(os.environ.get("LOOKBACK_SEC", "0.5"))
# Trailing buffer: keep capturing this long AFTER the stop keypress, so the final
# word — which the user often finishes saying as/just after they hit the hotkey —
# is included. Symmetric with LOOKBACK_SEC at the start.
TRAILING_SEC = float(os.environ.get("TRAILING_SEC", "0.5"))
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
# ready. main() pre-warms the model AND re-arms continuous capture immediately
# after the re-exec (killing the old capture sox first), so the next dictation
# still hits the warm path — the window is just the RAM-reset cadence, no longer a
# warm-vs-cold gate. Steady-state RSS with the model resident is ~1.4 GB.
RESTART_IDLE_SEC = int(os.environ.get("RESTART_IDLE_SEC", "2700"))  # default 45 min
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


class SessionManager:
    """Owns the lifecycle of one dictation at a time over the live ring buffer."""

    def __init__(self, ring, chunk_dir, transcribe_fn, type_fn, get_window_fn,
                 log_fn=lambda m: None, clock=time.monotonic, lookback_sec=LOOKBACK_SEC,
                 trailing_sec=TRAILING_SEC):
        self.ring = ring
        self.chunk_dir = chunk_dir
        self.transcribe_fn = transcribe_fn
        self.type_fn = type_fn
        self.get_window_fn = get_window_fn
        self.log = log_fn
        self.clock = clock
        self.lookback_sec = lookback_sec
        self.trailing_sec = trailing_sec
        self.active = Event()
        self._threads = []
        self.last_end = [clock()]

    def start(self):
        if self.active.is_set():
            return  # duplicate START ignored
        t_start = self.clock()
        session_q = self.ring.start_session(t_start, self.lookback_sec)
        window_id = self.get_window_fn()
        self.active.set()
        chunk_q = queue.Queue()
        chunker = Thread(target=self._chunker, args=(session_q, chunk_q), daemon=True)
        transcriber = Thread(target=self._transcriber, args=(chunk_q, window_id), daemon=True)
        chunker.start()
        transcriber.start()
        self._threads = [chunker, transcriber]
        self.log(f"session started (window {window_id})")

    def stop(self, trailing=None):
        if not self.active.is_set():
            return
        # Keep the tee armed for a trailing window so audio still arriving right
        # after the stop keypress (the tail of the final word) is captured. The
        # capture thread keeps appending to the live session queue during the
        # sleep; only then do we close it. trailing=0 stops immediately (pause).
        wait = self.trailing_sec if trailing is None else trailing
        if wait > 0:
            time.sleep(wait)
        self.ring.stop_session()          # disarms tee + pushes sentinel to session_q
        for t in self._threads:
            t.join(timeout=30)
        self._threads = []
        self.active.clear()
        self.last_end[0] = self.clock()
        self.log("session complete")

    def _chunker(self, session_q, chunk_q):
        def block_iter():
            while True:
                block = session_q.get()
                if block is None:         # sentinel from stop_session
                    return
                yield block
        num = 0
        for samples in split_blocks_to_chunks(
            block_iter(),
            silence_blocks=int(SILENCE_DUR / 0.05),
            silence_thresh=SILENCE_THRESH,
            min_chunk_samples=int(RATE * 0.3),
            max_samples=MAX_CHUNK_SEC * RATE,
        ):
            path = os.path.join(self.chunk_dir, f"chunk_{num}.wav")
            samples_to_wav(samples, path)
            chunk_q.put((path, num))
            num += 1
        chunk_q.put(None)                 # sentinel to transcriber

    def _transcriber(self, chunk_q, window_id):
        while True:
            item = chunk_q.get()
            if item is None:
                return
            path, num = item
            if not os.path.isfile(path):
                continue
            try:
                text = self.transcribe_fn(path)
            except Exception as e:
                self.log(f"transcribe error: {e}")
                text = ""
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
            # Skip empty / punctuation-only noise (whisper emits "." for silence).
            if not any(c.isalnum() for c in text):
                self.log(f"skipping non-speech: {text!r}")
                continue
            self.log(f"transcribed: {text}")
            try:
                self.type_fn(text, window_id)
            except Exception as e:
                self.log(f"type error: {e}")


def _read_exact(source, n):
    """Read exactly n bytes from source, looping over short reads (a pipe's
    .read(n) may return fewer bytes than asked). Returns None at true EOF."""
    buf = bytearray()
    while len(buf) < n:
        chunk = source.read(n - len(buf))
        if not chunk:          # b'' == EOF (e.g. capture sox died/was killed)
            return None
        buf.extend(chunk)
    return bytes(buf)


def capture_loop(source, ring, stop_event, clock=time.monotonic):
    """Read fixed BLOCK_BYTES frames from `source` (a file-like with .read) into
    the ring buffer until stop_event is set or the source hits EOF. The ring tees
    into the active session queue, if any. Short reads are coalesced into full
    blocks; a partial block at EOF is dropped."""
    while not stop_event.is_set():
        raw = _read_exact(source, BLOCK_BYTES)
        if raw is None:        # true EOF
            break
        ring.append(clock(), raw)


def control_loop(fifo_path, handlers, stop_event):
    """Read newline-delimited commands from the control FIFO and dispatch.
    Opened O_RDWR so the daemon always has a writer of its own -> reads block for
    data instead of hitting EOF, and external writers never block."""
    fd = os.open(fifo_path, os.O_RDWR)
    with os.fdopen(fd, "r", buffering=1) as f:
        while not stop_event.is_set():
            line = f.readline()
            if not line:
                continue
            cmd = line.strip().upper()
            handler = handlers.get(cmd)
            if handler:
                try:
                    handler()
                except Exception as e:
                    log(f"control handler error for {cmd}: {e}")
            elif cmd:
                log(f"unknown control command: {cmd!r}")


def collapse_repeated_segments(texts):
    """Join whisper segment texts, dropping empty ones and collapsing CONSECUTIVE
    duplicates. Whisper's decoding loop on non-speech emits the same sentence many
    times ("...close the door." x17); this collapses that to one. Distinct or
    non-adjacent segments are preserved. Joins with a single space (no doubles)."""
    out = []
    for t in texts:
        t = t.strip()
        if not t:
            continue
        if out and t == out[-1]:
            continue
        out.append(t)
    return " ".join(out).strip()


def make_transcribe_fn(model_holder):
    """Adapt the resident whisper model to SessionManager's transcribe_fn(path)."""
    def transcribe(path):
        model = model_holder[0]
        # vad_filter strips non-speech before decoding -> no hallucinated phrases
        # on silence/noise. condition_on_previous_text=False stops the decoder
        # feeding its own output back, which is what sustains repetition loops.
        segments, _ = model.transcribe(
            path, language="en", beam_size=1, temperature=0,
            vad_filter=True, condition_on_previous_text=False,
        )
        return collapse_repeated_segments(s.text for s in segments)
    return transcribe


def type_into_window(text, window_id):
    """SessionManager's type_fn: focus the target window, then type the text.
    windowactivate is more reliable than --window (synthetic events many apps
    ignore)."""
    if window_id:
        subprocess.run(["xdotool", "windowactivate", "--sync", window_id], timeout=2)
    subprocess.run(
        ["xdotool", "type", "--clearmodifiers", "--delay", "0", "--", text + " "],
        timeout=5,
    )


def split_blocks_to_chunks(blocks, *, silence_blocks, silence_thresh,
                           min_chunk_samples, max_samples):
    """Consume raw byte blocks; yield chunks as sample lists, splitting on a
    natural silence pause or the MAX hard cap. Mirrors the original reader_loop
    splitting. Flushes the remainder at end-of-stream if >= min_chunk_samples."""
    audio_buf = []
    silent_blocks = 0
    has_speech = False
    for raw in blocks:
        if not raw:
            break
        n = len(raw) // SAMPWIDTH
        samples = struct.unpack(f"<{n}h", raw[:n * SAMPWIDTH])
        audio_buf.extend(samples)

        if rms(samples) < silence_thresh:
            silent_blocks += 1
        else:
            has_speech = True
            silent_blocks = 0

        should_split = False
        if has_speech and silent_blocks >= silence_blocks and len(audio_buf) >= min_chunk_samples:
            should_split = True
        elif len(audio_buf) >= max_samples:
            should_split = True

        if should_split:
            yield audio_buf
            audio_buf = []
            silent_blocks = 0
            has_speech = False

    if audio_buf and len(audio_buf) >= min_chunk_samples:
        yield audio_buf


def main():
    model_name = os.environ.get("WHISPER_MODEL", "medium.en")
    device = os.environ.get("WHISPER_DEVICE", "cuda")
    compute_type = "int8" if device == "cuda" else "float32"

    os.makedirs(CHUNK_DIR, exist_ok=True)

    # Create the control FIFO (start/stop/pause/resume commands from the scripts).
    if os.path.exists(CONTROL_FIFO) and not stat_is_fifo(CONTROL_FIFO):
        os.remove(CONTROL_FIFO)
    if not os.path.exists(CONTROL_FIFO):
        os.mkfifo(CONTROL_FIFO)

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

    # --- Continuous capture: the daemon owns the mic and fills sm.ring 24/7. ---
    # Defined before the watchdog because the watchdog's pre-execv path calls
    # stop_capture() (the capture sox must be killed before execv, else it is
    # orphaned writing to a dead pipe).
    capture_stop = Event()
    capture_proc = [None]   # persistent sox subprocess (list so closures can swap it)

    sm = SessionManager(
        ring=RingBuffer(RING_BLOCKS),
        chunk_dir=CHUNK_DIR,
        transcribe_fn=make_transcribe_fn(model_holder),
        type_fn=type_into_window,
        get_window_fn=get_target_window,
        log_fn=log,
        lookback_sec=LOOKBACK_SEC,
    )

    def start_capture():
        """Spawn the persistent sox + a supervisor thread reading it into sm.ring."""
        if capture_proc[0] is not None:
            return
        src = os.environ.get("HOTMIC_SOURCE")
        argv = (["sox", "-q", "-t", "alsa", src] + CAPTURE_CMD_DEFAULT[3:]) if src else CAPTURE_CMD_DEFAULT
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                                stderr=open(LOG_FILE, "a"), bufsize=0)
        capture_proc[0] = proc
        capture_stop.clear()
        Thread(target=_capture_supervisor, args=(proc,), daemon=True).start()
        log("capture started (mic armed)")

    def _capture_supervisor(proc):
        try:
            capture_loop(proc.stdout, sm.ring, capture_stop)
        finally:
            try:
                proc.wait(timeout=2)   # reap so we don't leave a <defunct> child
            except Exception:
                pass
        # Returned: either we asked it to stop, or sox died on its own.
        if not capture_stop.is_set() and not daemon_stop.is_set():
            log("capture sox ended unexpectedly; respawning")
            capture_proc[0] = None
            time.sleep(0.5)
            start_capture()

    def stop_capture():
        capture_stop.set()
        proc = capture_proc[0]
        capture_proc[0] = None
        if proc is not None:
            try:
                proc.kill()   # EOF on proc.stdout unblocks capture_loop's read()
            except Exception:
                pass

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
                # Kill the persistent capture sox first — execv would otherwise
                # orphan it writing to a dead pipe. Keep whisper.ready + control
                # FIFO in place so the scripts keep seeing a ready daemon. execv
                # replaces this whole process image and re-runs main() fresh,
                # inheriting the environment (LD_LIBRARY_PATH, WHISPER_*, ...).
                stop_capture()
                os.execv(sys.executable, [sys.executable, os.path.abspath(__file__)])
                return  # unreachable if execv succeeds

    watchdog = Thread(target=idle_watchdog, daemon=True)
    watchdog.start()

    # Pre-warm the model eagerly in the background so the daemon is ready to
    # transcribe the instant the next dictation lands — no first-session reload
    # delay. Runs on every fresh start AND after every idle re-exec (execv re-runs
    # main()). Background load keeps readiness immediate. Holding the model
    # resident is ~1.4 GB RSS — acceptable now that runaway growth is bounded.
    Thread(target=load_model, daemon=True).start()

    # --- Session + pause control handlers (operate on the continuous capture) ---
    def do_start():
        in_session.set()
        if capture_proc[0] is None:    # armed from paused -> cold path, no lookback
            if os.path.exists(PAUSED_FLAG):
                os.remove(PAUSED_FLAG)
            start_capture()
        sm.start()

    def do_stop():
        sm.stop()
        last_session_end[0] = sm.last_end[0]
        in_session.clear()

    def do_pause():
        if sm.active.is_set():
            sm.stop(trailing=0)   # release the mic immediately on pause
            in_session.clear()
        stop_capture()
        open(PAUSED_FLAG, "w").close()
        log("paused (mic released)")

    def do_resume():
        if os.path.exists(PAUSED_FLAG):
            os.remove(PAUSED_FLAG)
        start_capture()
        log("resumed (mic armed)")

    handlers = {"START": do_start, "STOP": do_stop, "PAUSE": do_pause, "RESUME": do_resume}

    # Arm capture now unless we were paused before a re-exec.
    if not os.path.exists(PAUSED_FLAG):
        start_capture()

    control = Thread(target=control_loop, args=(CONTROL_FIFO, handlers, daemon_stop), daemon=True)
    control.start()

    # Signal readiness — control FIFO exists, capture armed, daemon accepting.
    open(READY_FILE, "w").close()
    log(f"daemon ready, pre-warming model + continuous capture "
        f"(idle restart: {RESTART_IDLE_SEC}s)")

    # The watchdog, control, capture and session threads do the work; the main
    # thread just idles until shutdown.
    while not daemon_stop.is_set():
        daemon_stop.wait(1.0)

    stop_capture()
    try:
        os.remove(READY_FILE)
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
