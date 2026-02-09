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
    model_name = os.environ.get("WHISPER_MODEL", "tiny")
    device = os.environ.get("WHISPER_DEVICE", "cuda")
    fp16 = device == "cuda"

    try:
        log(f"loading model={model_name} device={device}")
        import whisper
        model = whisper.load_model(model_name, device=device)
        log("model loaded, ready for chunks")
    except Exception as e:
        if device != "cpu":
            log(f"CUDA failed ({e}), falling back to CPU")
            try:
                model = whisper.load_model(model_name, device="cpu")
                fp16 = False
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
                        result = model.transcribe(
                            chunk_path,
                            language="en",
                            temperature=0,
                            beam_size=1,
                            best_of=1,
                            fp16=fp16,
                        )
                        text = result.get("text", "").strip()
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
