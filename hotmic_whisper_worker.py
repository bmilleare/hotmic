#!/usr/bin/env python3
"""
Continuous audio reader, splitter, and transcriber.

Sox pipes raw PCM into stdin. This process:
1. Immediately starts reading audio (keeps the pipe drained)
2. Loads the whisper model in the background
3. Splits audio on silence or max duration
4. Transcribes each chunk in a background thread
5. Types results into the active window via xdotool
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
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] whisper-worker: {msg}\n")


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


def reader_loop(chunk_queue, stop_event):
    """Read continuous PCM from stdin and split into chunks. Runs immediately."""
    chunk_num = 0
    audio_buf = []
    silent_blocks = 0
    has_speech = False
    max_samples = MAX_CHUNK_SEC * RATE

    log(f"reading audio (max_chunk={MAX_CHUNK_SEC}s, silence_dur={SILENCE_DUR}s, silence_thresh={SILENCE_THRESH})")

    stdin = sys.stdin.buffer
    try:
        while not stop_event.is_set():
            raw = stdin.read(BLOCK_BYTES)
            if not raw:
                break  # sox closed pipe (recording stopped)

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


def transcriber_loop(model_holder, chunk_queue, stop_event):
    """Background thread: waits for model, then transcribes chunks and types results."""
    # Wait for model to be loaded.
    # Don't bail on stop_event if chunks are pending — a short recording may
    # stop before the model finishes loading, but we still need to transcribe.
    deadline = time.monotonic() + 60  # absolute max wait for model
    while not model_holder:
        if stop_event.is_set() and not chunk_queue:
            return  # stopped with nothing to transcribe
        if time.monotonic() > deadline:
            log("model load timeout, giving up")
            return
        time.sleep(0.1)

    if not model_holder:
        return

    model = model_holder[0]
    log("transcriber ready")

    while not stop_event.is_set() or chunk_queue:
        if not chunk_queue:
            stop_event.wait(0.05)
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

    # Graceful shutdown
    stop_event = Event()
    def handle_signal(signum, frame):
        stop_event.set()
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Shared model holder (list so transcriber thread can see when it's loaded)
    model_holder = []
    chunk_queue = deque()

    # Start reader thread IMMEDIATELY — keeps the pipe drained while model loads
    reader = Thread(target=reader_loop, args=(chunk_queue, stop_event), daemon=True)
    reader.start()

    # Start transcriber thread (waits for model internally)
    transcriber = Thread(target=transcriber_loop, args=(model_holder, chunk_queue, stop_event), daemon=True)
    transcriber.start()

    # Load model in a background thread so the main thread isn't blocked in
    # a C extension call (which can't be interrupted by signals or timeouts).
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

    # Wait for model to load (up to 45s)
    loader.join(timeout=45)
    if not model_holder:
        log("FATAL: model load timed out or failed")
        stop_event.set()
        sys.exit(1)

    # Signal readiness
    open(READY_FILE, "w").close()

    # Wait for reader to finish (pipe closed = sox died or was killed).
    # Reader may already be done if the user stopped during model loading.
    reader.join()

    # Wait for transcriber to drain remaining chunks before signalling stop.
    drain_deadline = time.monotonic() + 30
    while chunk_queue and time.monotonic() < drain_deadline:
        time.sleep(0.1)

    log("waiting for transcription to complete...")
    stop_event.set()
    transcriber.join(timeout=15)

    # Cleanup
    try:
        os.remove(READY_FILE)
    except OSError:
        pass
    log("worker exiting")


if __name__ == "__main__":
    main()
