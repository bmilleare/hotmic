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
import subprocess
import traceback
from collections import deque
from threading import Thread, Event

DIR = "/tmp/hotmic"
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

    # Flush remaining audio
    if audio_buf and has_speech and len(audio_buf) >= MIN_CHUNK_SAMPLES:
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

        if not text:
            log("empty transcription, skipping")
            continue

        log(f"transcribed: {text}")
        try:
            cmd = ["xdotool", "type", "--clearmodifiers", "--delay", "0"]
            if window_id:
                cmd.extend(["--window", window_id])
            cmd.extend(["--", text + " "])
            subprocess.run(cmd, timeout=5)
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

    # Model holder — shared between sessions, populated by loader thread
    model_holder = []

    # Load model in a background thread so the daemon can accept audio immediately.
    # The first session's transcriber waits for the model; subsequent sessions have it ready.
    def load_model():
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

    loader = Thread(target=load_model, daemon=True)
    loader.start()

    # Signal readiness — FIFO exists, daemon is accepting sessions.
    # Model may still be loading; the transcriber waits for it.
    open(READY_FILE, "w").close()
    log("daemon ready, accepting audio (model loading in background)")

    # === Main session loop ===
    while not daemon_stop.is_set():
        # Open FIFO for reading — blocks until sox opens it for writing
        try:
            fifo = open(FIFO_PATH, "rb")
        except OSError as e:
            if daemon_stop.is_set():
                break
            log(f"FIFO open error: {e}")
            time.sleep(0.5)
            continue

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

        log("session complete")

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
