#!/usr/bin/env python3
"""Persistent whisper worker — loads model once, transcribes chunks from a FIFO."""

import os
import sys
import signal
import traceback

DIR = "/tmp/hotmic"
FIFO_PATH = f"{DIR}/whisper.fifo"
READY_FILE = f"{DIR}/whisper.ready"
LOG_FILE = f"{DIR}/hotmic.log"


def log(msg):
    from datetime import datetime
    with open(LOG_FILE, "a") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] whisper-worker: {msg}\n")


def main():
    model_name = os.environ.get("WHISPER_MODEL", "medium.en")
    device = os.environ.get("WHISPER_DEVICE", "cuda")
    # int8 for CUDA (Pascal-friendly, no Tensor cores needed), float32 for CPU
    compute_type = "int8" if device == "cuda" else "float32"

    try:
        log(f"loading model={model_name} device={device} compute_type={compute_type}")
        from faster_whisper import WhisperModel
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        log("model loaded, ready for chunks")
    except Exception as e:
        if device != "cpu":
            log(f"CUDA failed ({e}), falling back to CPU")
            try:
                model = WhisperModel(model_name, device="cpu", compute_type="float32")
                log("model loaded on CPU, ready for chunks")
            except Exception as e2:
                log(f"FATAL: failed to load model on CPU: {e2}")
                traceback.print_exc(file=open(LOG_FILE, "a"))
                sys.exit(1)
        else:
            log(f"FATAL: failed to load model: {e}")
            traceback.print_exc(file=open(LOG_FILE, "a"))
            sys.exit(1)

    # Signal readiness to bash script
    open(READY_FILE, "w").close()

    # Graceful shutdown
    running = True
    def handle_signal(signum, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    while running:
        try:
            with open(FIFO_PATH, "r") as fifo:
                for line in fifo:
                    chunk_path = line.strip()
                    if not chunk_path or not os.path.isfile(chunk_path):
                        continue

                    size = os.path.getsize(chunk_path)
                    log(f"transcribing {chunk_path} ({size} bytes)")

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
                        os.remove(chunk_path)

                    if not text:
                        log("empty transcription, skipping")
                        continue

                    log(f"transcribed: {text}")

                    # Write result to a .txt file for the bash script to pick up
                    result_path = chunk_path.rsplit(".", 1)[0] + ".txt"
                    with open(result_path, "w") as f:
                        f.write(text)
        except OSError:
            if running:
                continue
            break

    # Cleanup
    try:
        os.remove(READY_FILE)
    except OSError:
        pass
    log("worker exiting")


if __name__ == "__main__":
    main()
