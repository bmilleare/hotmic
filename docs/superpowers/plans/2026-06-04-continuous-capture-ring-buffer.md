# Continuous-Capture Ring Buffer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the cold-start gap that drops the first ~2–4 s of speech on the first dictation after a long idle, by having the daemon continuously capture the mic into a RAM ring buffer and treating each dictation as a `[t_start − lookback, t_stop]` window over that always-live stream.

**Architecture:** Single-file daemon keeps single-file deploy. Pure, testable units (`RingBuffer`, `split_blocks_to_chunks`, `capture_loop`, `SessionManager`) live at module level in `hotmic_whisper_worker.py` and are imported directly by pytest. A persistent `sox` subprocess feeds a capture thread → ring buffer; a control thread reads `START`/`STOP`/`PAUSE`/`RESUME` from a control FIFO. `start.sh`/`stop.sh` stop spawning `sox` and instead signal the daemon; new `hotmic_pause.sh` toggles capture.

**Tech Stack:** Python 3.10 stdlib (`threading`, `queue`, `collections.deque`, `struct`, `wave`, `subprocess`), faster_whisper (lazy import, unchanged), pytest 8.x, sox, bash.

**Spec:** `docs/superpowers/specs/2026-06-04-continuous-capture-ring-buffer-design.md`

---

## File Structure

- **Modify** `hotmic_whisper_worker.py` — add `RingBuffer`, `split_blocks_to_chunks`, `capture_loop`, `SessionManager`, control-FIFO + pause handling; rewire `main()`. Pure logic at module level; lazy `faster_whisper` import stays inside `load_model`.
- **Create** `tests/test_worker.py` — unit + pipeline tests (no GPU; import the module directly).
- **Create** `tests/__init__.py` — empty (package marker, optional).
- **Modify** `hotmic_start.sh` — whisper branch signals `START` instead of launching `sox`.
- **Modify** `hotmic_stop.sh` — signal `STOP` instead of killing the per-session `sox`.
- **Create** `hotmic_pause.sh` — toggle `PAUSE`/`RESUME`.

**Module constants (add near the existing audio-format constants):**

```python
import queue
from threading import Lock

RING_SECONDS = float(os.environ.get("RING_SECONDS", "10"))
LOOKBACK_SEC = float(os.environ.get("LOOKBACK_SEC", "2.0"))
RING_BLOCKS = max(1, int(RING_SECONDS / 0.05))   # 50ms blocks -> 200 for 10s
CONTROL_FIFO = f"{DIR}/control.fifo"
PAUSED_FLAG = f"{DIR}/paused"
CAPTURE_CMD_DEFAULT = ["sox", "-q", "-d", "-t", "raw",
                       "-r", str(RATE), "-c", str(CHANNELS), "-b", "16",
                       "-e", "signed-integer", "-"]
```

(`RATE`, `CHANNELS`, `SAMPWIDTH`, `BLOCK_SAMPLES`, `BLOCK_BYTES`, `rms`, `samples_to_wav` already exist and are reused unchanged. `RESTART_IDLE_SEC` default changes from `1200` to `2700`.)

---

### Task 1: RingBuffer with atomic lookback + tee

**Files:**
- Modify: `hotmic_whisper_worker.py` (add `RingBuffer` class)
- Test: `tests/test_worker.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_worker.py
import sys, os, struct
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hotmic_whisper_worker as w

BB = w.BLOCK_BYTES

def _blk(byte=0):
    return bytes([byte]) * BB

def test_ring_evicts_to_maxlen():
    ring = w.RingBuffer(maxlen=3)
    for i in range(5):
        ring.append(float(i), _blk(i % 256))
    snap = ring.snapshot()            # list[(ts, block)] oldest->newest
    assert [ts for ts, _ in snap] == [2.0, 3.0, 4.0]

def test_start_session_seeds_lookback_only():
    ring = w.RingBuffer(maxlen=200)
    # blocks at ts 0.0, 0.5, 1.0, ..., 5.0
    for i in range(11):
        ring.append(i * 0.5, _blk(i))
    q = ring.start_session(t_start=5.0, lookback_sec=2.0)   # keep ts >= 3.0
    blocks = _drain(q)
    assert len(blocks) == 5            # ts 3.0,3.5,4.0,4.5,5.0

def test_tee_after_start_no_gap_no_dup():
    ring = w.RingBuffer(maxlen=200)
    ring.append(0.0, _blk(1))          # pre-session, before lookback window
    ring.append(10.0, _blk(2))         # in lookback window
    q = ring.start_session(t_start=10.0, lookback_sec=2.0)
    ring.append(10.05, _blk(3))        # arrives after arm -> must be teed exactly once
    ring.append(10.10, _blk(4))
    blocks = _drain(q)
    assert blocks == [_blk(2), _blk(3), _blk(4)]   # seeded + teed, ordered, no dup

def test_stop_session_pushes_sentinel_and_disarms():
    ring = w.RingBuffer(maxlen=200)
    ring.append(1.0, _blk(1))
    q = ring.start_session(t_start=1.0, lookback_sec=2.0)
    ring.stop_session()
    ring.append(2.0, _blk(9))          # must NOT be teed after stop
    got = []
    while True:
        item = q.get(timeout=1)
        if item is None:               # sentinel
            break
        got.append(item)
    assert got == [_blk(1)]

def _drain(q):
    """Drain a queue.Queue of blocks up to a sentinel or empty."""
    import queue as _q
    out = []
    while True:
        try:
            item = q.get_nowait()
        except _q.Empty:
            break
        if item is None:
            break
        out.append(item)
    return out
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cd ~/dev/hotmic && python3 -m pytest tests/test_worker.py -k ring -v`
Expected: FAIL — `AttributeError: module 'hotmic_whisper_worker' has no attribute 'RingBuffer'`

- [ ] **Step 3: Implement `RingBuffer`**

```python
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
```

- [ ] **Step 4: Run tests, verify pass**

Run: `python3 -m pytest tests/test_worker.py -k ring -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add tests/test_worker.py hotmic_whisper_worker.py
git commit -m "feat: RingBuffer with atomic lookback snapshot + session tee"
```

---

### Task 2: `split_blocks_to_chunks` chunker (extracted from `reader_loop`)

**Files:**
- Modify: `hotmic_whisper_worker.py` (add generator; reuses existing `rms`)
- Test: `tests/test_worker.py`

- [ ] **Step 1: Write the failing tests**

```python
def _speech_blk():
    # amplitude 10000 -> rms ~0.305, above SILENCE_THRESH (0.03)
    return struct.pack(f"<{w.BLOCK_SAMPLES}h", *([10000] * w.BLOCK_SAMPLES))

def _silence_blk():
    return b"\x00" * w.BLOCK_BYTES

def test_chunker_splits_on_silence():
    # 20 speech blocks then SILENCE_BLOCKS silent blocks -> one chunk
    sb = int(w.SILENCE_DUR / 0.05)     # SILENCE_BLOCKS (16 by default)
    blocks = [_speech_blk()] * 20 + [_silence_blk()] * sb
    chunks = list(w.split_blocks_to_chunks(
        iter(blocks),
        silence_blocks=sb,
        silence_thresh=w.SILENCE_THRESH,
        min_chunk_samples=int(w.RATE * 0.3),
        max_samples=w.MAX_CHUNK_SEC * w.RATE,
    ))
    assert len(chunks) == 1
    assert len(chunks[0]) == (20 + sb) * w.BLOCK_SAMPLES

def test_chunker_hard_caps_at_max_samples():
    # continuous speech longer than max -> at least 2 chunks, none over max
    max_samples = 2 * w.RATE           # tiny cap for the test: 2s
    nblocks = int(5 * w.RATE / w.BLOCK_SAMPLES)   # ~5s of speech
    chunks = list(w.split_blocks_to_chunks(
        iter([_speech_blk()] * nblocks),
        silence_blocks=9999,           # never silence-split
        silence_thresh=w.SILENCE_THRESH,
        min_chunk_samples=int(w.RATE * 0.3),
        max_samples=max_samples,
    ))
    assert len(chunks) >= 2
    assert all(len(c) <= max_samples for c in chunks)

def test_chunker_flushes_remainder():
    # speech with no trailing silence -> flushed at end if >= min
    blocks = [_speech_blk()] * 10      # 8000 samples > min 4800
    chunks = list(w.split_blocks_to_chunks(
        iter(blocks),
        silence_blocks=16,
        silence_thresh=w.SILENCE_THRESH,
        min_chunk_samples=int(w.RATE * 0.3),
        max_samples=w.MAX_CHUNK_SEC * w.RATE,
    ))
    assert len(chunks) == 1
    assert len(chunks[0]) == 10 * w.BLOCK_SAMPLES

def test_chunker_drops_subminimum_remainder():
    blocks = [_speech_blk()] * 2       # 1600 samples < min 4800
    chunks = list(w.split_blocks_to_chunks(
        iter(blocks),
        silence_blocks=16,
        silence_thresh=w.SILENCE_THRESH,
        min_chunk_samples=int(w.RATE * 0.3),
        max_samples=w.MAX_CHUNK_SEC * w.RATE,
    ))
    assert chunks == []
```

- [ ] **Step 2: Run tests, verify fail**

Run: `python3 -m pytest tests/test_worker.py -k chunker -v`
Expected: FAIL — `AttributeError: ... 'split_blocks_to_chunks'`

- [ ] **Step 3: Implement the generator (logic lifted verbatim from `reader_loop`)**

```python
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
```

- [ ] **Step 4: Run tests, verify pass**

Run: `python3 -m pytest tests/test_worker.py -k chunker -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add tests/test_worker.py hotmic_whisper_worker.py
git commit -m "feat: split_blocks_to_chunks chunker extracted from reader_loop"
```

---

### Task 3: `capture_loop` (injectable source → ring buffer)

**Files:**
- Modify: `hotmic_whisper_worker.py`
- Test: `tests/test_worker.py`

- [ ] **Step 1: Write the failing tests**

```python
import io

def test_capture_loop_fills_ring_and_stops_on_eof():
    from threading import Event
    ring = w.RingBuffer(maxlen=200)
    n = 5
    src = io.BytesIO(b"".join(_silence_blk() for _ in range(n)))
    stop = Event()
    ticks = iter([float(i) for i in range(100)])
    w.capture_loop(src, ring, stop, clock=lambda: next(ticks))
    assert len(ring.snapshot()) == n     # stopped at EOF

def test_capture_loop_tees_into_active_session():
    from threading import Event
    ring = w.RingBuffer(maxlen=200)
    # seed one block, start a session, then capture more
    ring.append(0.0, _silence_blk())
    q = ring.start_session(t_start=0.0, lookback_sec=2.0)
    src = io.BytesIO(b"".join(_speech_blk() for _ in range(3)))
    stop = Event()
    ticks = iter([0.01, 0.02, 0.03])
    w.capture_loop(src, ring, stop, clock=lambda: next(ticks))
    drained = _drain(q)
    assert len(drained) == 1 + 3         # seeded silence + 3 teed speech
```

- [ ] **Step 2: Run tests, verify fail**

Run: `python3 -m pytest tests/test_worker.py -k capture -v`
Expected: FAIL — `AttributeError: ... 'capture_loop'`

- [ ] **Step 3: Implement**

```python
def capture_loop(source, ring, stop_event, clock=time.monotonic):
    """Read BLOCK_BYTES at a time from `source` (a file-like with .read) into the
    ring buffer until stop_event is set or the source hits EOF. The ring tees into
    the active session queue, if any. Partial trailing reads are dropped."""
    while not stop_event.is_set():
        raw = source.read(BLOCK_BYTES)
        if not raw or len(raw) < BLOCK_BYTES:
            break
        ring.append(clock(), raw)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `python3 -m pytest tests/test_worker.py -k capture -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add tests/test_worker.py hotmic_whisper_worker.py
git commit -m "feat: capture_loop reads injectable source into ring buffer"
```

---

### Task 4: `SessionManager` (start/stop → chunker + transcriber over the session queue)

**Files:**
- Modify: `hotmic_whisper_worker.py`
- Test: `tests/test_worker.py`

`SessionManager` wires a session: on `start()` it asks the ring for a seeded
queue, then runs the chunker (producing WAV chunk files) and a transcriber that
calls injected `transcribe_fn(path) -> str` and `type_fn(text, window_id)`.
Injection keeps it GPU-free and deterministic in tests.

- [ ] **Step 1: Write the failing pipeline test**

```python
def test_session_manager_includes_lookback_and_types(tmp_path, monkeypatch):
    monkeypatch.setenv("HOTMIC_DIR", str(tmp_path))
    (tmp_path / "chunks").mkdir()
    typed = []
    # fake transcribe: report the chunk's sample count so we can assert lookback
    def fake_transcribe(path):
        import wave
        with wave.open(path, "rb") as wf:
            return f"n={wf.getnframes()}"
    sm = w.SessionManager(
        ring=w.RingBuffer(maxlen=200),
        chunk_dir=str(tmp_path / "chunks"),
        transcribe_fn=fake_transcribe,
        type_fn=lambda text, wid: typed.append(text),
        get_window_fn=lambda: "win",
        clock=_FakeClock(),
        lookback_sec=2.0,
    )
    # 2s of lookback already in the ring before start
    sb = int(w.SILENCE_DUR / 0.05)
    for i in range(40):                       # 40 * 50ms = 2.0s
        sm.ring.append(8.0 + i * 0.05, _speech_blk())
    sm.clock.now = 10.0
    sm.start()                                # t_start=10.0 -> lookback grabs the 40 blocks
    # speak a bit more then go silent to force a chunk
    for i in range(20):
        sm.ring.append(10.0 + i * 0.05, _speech_blk())
    for i in range(sb):
        sm.ring.append(11.0 + i * 0.05, _silence_blk())
    sm.stop()                                 # drains + joins
    assert typed, "expected at least one typed chunk"
    # first chunk must contain >= the lookback (40 blocks) of audio
    first_n = int(typed[0].split("=")[1])
    assert first_n >= 40 * w.BLOCK_SAMPLES

class _FakeClock:
    def __init__(self): self.now = 0.0
    def __call__(self): return self.now
```

- [ ] **Step 2: Run test, verify fail**

Run: `python3 -m pytest tests/test_worker.py -k session_manager -v`
Expected: FAIL — `AttributeError: ... 'SessionManager'`

- [ ] **Step 3: Implement `SessionManager`**

```python
class SessionManager:
    """Owns the lifecycle of one dictation at a time over the live ring buffer."""

    def __init__(self, ring, chunk_dir, transcribe_fn, type_fn, get_window_fn,
                 log_fn=lambda m: None, clock=time.monotonic, lookback_sec=LOOKBACK_SEC):
        self.ring = ring
        self.chunk_dir = chunk_dir
        self.transcribe_fn = transcribe_fn
        self.type_fn = type_fn
        self.get_window_fn = get_window_fn
        self.log = log_fn
        self.clock = clock
        self.lookback_sec = lookback_sec
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
        chunker.start(); transcriber.start()
        self._threads = [chunker, transcriber]
        self.log(f"session started (window {window_id})")

    def stop(self):
        if not self.active.is_set():
            return
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
                try: os.remove(path)
                except OSError: pass
            if not any(c.isalnum() for c in text):
                self.log(f"skipping non-speech: {text!r}")
                continue
            self.log(f"transcribed: {text}")
            try:
                self.type_fn(text, window_id)
            except Exception as e:
                self.log(f"type error: {e}")
```

- [ ] **Step 4: Run test, verify pass**

Run: `python3 -m pytest tests/test_worker.py -k session_manager -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_worker.py hotmic_whisper_worker.py
git commit -m "feat: SessionManager runs chunker+transcriber over live ring buffer"
```

---

### Task 5: Real model transcribe + xdotool type adapters

**Files:**
- Modify: `hotmic_whisper_worker.py`
- Test: none (thin adapters over external deps; covered by manual verification)

These adapt the resident model and `xdotool` to the `transcribe_fn` / `type_fn`
interfaces `SessionManager` expects. The transcribe params and the
windowactivate→type sequence are copied verbatim from today's `transcriber_loop`.

- [ ] **Step 1: Add adapters**

```python
def make_transcribe_fn(model_holder):
    def transcribe(path):
        model = model_holder[0]
        segments, _ = model.transcribe(path, language="en", beam_size=1, temperature=0)
        return " ".join(s.text for s in segments).strip()
    return transcribe

def type_into_window(text, window_id):
    if window_id:
        subprocess.run(["xdotool", "windowactivate", "--sync", window_id], timeout=2)
    subprocess.run(
        ["xdotool", "type", "--clearmodifiers", "--delay", "0", "--", text + " "],
        timeout=5,
    )
```

- [ ] **Step 2: Verify module still imports**

Run: `python3 -c "import hotmic_whisper_worker as w; print(callable(w.make_transcribe_fn), callable(w.type_into_window))"`
Expected: `True True`

- [ ] **Step 3: Commit**

```bash
git add hotmic_whisper_worker.py
git commit -m "feat: model transcribe + xdotool type adapters for SessionManager"
```

---

### Task 6: Control FIFO + pause handling + `main()` rewire

**Files:**
- Modify: `hotmic_whisper_worker.py` (replace `reader_loop`/`transcriber_loop`/session loop in `main()`; keep `load_model`, pre-warm, watchdog; bump `RESTART_IDLE_SEC` default to `2700`)
- Test: `tests/test_worker.py` (control-dispatch unit test over a real temp FIFO)

- [ ] **Step 1: Write the failing control-dispatch test**

```python
def test_control_thread_dispatches_commands(tmp_path):
    import os, time
    from threading import Event
    fifo = str(tmp_path / "control.fifo")
    os.mkfifo(fifo)
    seen = []
    stop = Event()
    handlers = {
        "START": lambda: seen.append("START"),
        "STOP": lambda: seen.append("STOP"),
        "PAUSE": lambda: seen.append("PAUSE"),
        "RESUME": lambda: seen.append("RESUME"),
    }
    t = Thread(target=w.control_loop, args=(fifo, handlers, stop), daemon=True)
    t.start()
    with open(fifo, "w") as f:
        f.write("START\nSTOP\nPAUSE\nRESUME\n"); f.flush()
    deadline = time.time() + 3
    while len(seen) < 4 and time.time() < deadline:
        time.sleep(0.02)
    stop.set()
    assert seen == ["START", "STOP", "PAUSE", "RESUME"]
```

- [ ] **Step 2: Run, verify fail**

Run: `python3 -m pytest tests/test_worker.py -k control -v`
Expected: FAIL — `AttributeError: ... 'control_loop'`

- [ ] **Step 3: Implement `control_loop`**

```python
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
```

- [ ] **Step 4: Run, verify pass**

Run: `python3 -m pytest tests/test_worker.py -k control -v`
Expected: PASS

- [ ] **Step 5: Bump idle default + ensure the control FIFO exists**

Change the existing line:

```python
RESTART_IDLE_SEC = int(os.environ.get("RESTART_IDLE_SEC", "2700"))  # default 45 min
```

In `main()`, alongside the existing audio-FIFO setup, create the control FIFO:

```python
    for fifo in (FIFO_PATH, CONTROL_FIFO):
        if os.path.exists(fifo) and not stat_is_fifo(fifo):
            os.remove(fifo)
        if not os.path.exists(fifo):
            os.mkfifo(fifo)
```

- [ ] **Step 6: Rewire `main()` — capture + control + session, replacing the FIFO session loop**

**Ordering requirement:** the capture primitives (`capture_stop`, `capture_proc`, `start_capture`, `_capture_supervisor`, `stop_capture`) must be defined **before** the watchdog thread is started, because the watchdog's pre-`execv` path calls `stop_capture()`. Concretely: move the capture-primitive definitions to just **above** the existing `watchdog = Thread(target=idle_watchdog, ...); watchdog.start()` lines, define `sm`/handlers/control after the pre-warm, and add `stop_capture()` immediately before the `os.execv(...)` call inside `idle_watchdog`.

Replace the body from the `# === Main session loop ===` comment through the end of the `while not daemon_stop.is_set():` loop with the continuous-capture wiring. Keep everything above it (signals, `model_holder`, `load_model`, `last_session_end`, `in_session`, `dictation_active`, the watchdog definition) and the eager pre-warm. Delete the now-unused `reader_loop` and `transcriber_loop` functions. New wiring:

```python
    # Continuous capture state
    capture_stop = Event()
    capture_proc = [None]   # the persistent sox subprocess

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
        """Spawn the persistent sox and a thread reading it into sm.ring."""
        if capture_proc[0] is not None:
            return
        cmd = os.environ.get("HOTMIC_SOURCE")
        argv = (["sox", "-q", "-t", "alsa", cmd] + CAPTURE_CMD_DEFAULT[3:]) if cmd else CAPTURE_CMD_DEFAULT
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                                stderr=open(LOG_FILE, "a"), bufsize=0)
        capture_proc[0] = proc
        capture_stop.clear()
        Thread(target=_capture_supervisor, args=(proc,), daemon=True).start()
        log("capture started (mic armed)")

    def _capture_supervisor(proc):
        # Re-read from sox; if it dies while still armed, respawn with backoff.
        capture_loop(proc.stdout, sm.ring, capture_stop)
        if not capture_stop.is_set() and not daemon_stop.is_set():
            log("capture sox ended unexpectedly; respawning")
            try: proc.kill()
            except Exception: pass
            capture_proc[0] = None
            time.sleep(0.5)
            start_capture()

    def stop_capture():
        capture_stop.set()
        proc = capture_proc[0]
        capture_proc[0] = None
        if proc is not None:
            try: proc.kill()
            except Exception: pass

    def do_start():
        in_session.set()
        if capture_proc[0] is None:          # armed-from-paused -> cold path, no lookback
            os.remove(PAUSED_FLAG) if os.path.exists(PAUSED_FLAG) else None
            start_capture()
        sm.start()

    def do_stop():
        sm.stop()
        last_session_end[0] = sm.last_end[0]
        in_session.clear()

    def do_pause():
        if sm.active.is_set():
            sm.stop(); in_session.clear()
        stop_capture()
        open(PAUSED_FLAG, "w").close()
        log("paused (mic released)")

    def do_resume():
        if os.path.exists(PAUSED_FLAG):
            os.remove(PAUSED_FLAG)
        start_capture()
        log("resumed (mic armed)")

    handlers = {"START": do_start, "STOP": do_stop, "PAUSE": do_pause, "RESUME": do_resume}

    # Arm capture now unless we were paused before a re-exec
    if not os.path.exists(PAUSED_FLAG):
        start_capture()

    control = Thread(target=control_loop, args=(CONTROL_FIFO, handlers, daemon_stop), daemon=True)
    control.start()

    # Idle the main thread until shutdown (watchdog + control + capture run as threads)
    while not daemon_stop.is_set():
        daemon_stop.wait(1.0)

    stop_capture()
```

In the watchdog, **kill the capture sox before `os.execv`** so it is not orphaned. Add immediately before the `os.execv(...)` call:

```python
                stop_capture()
```

- [ ] **Step 7: Pipeline smoke test (fake source + fake transcribe via env), run once**

Run:
```bash
cd ~/dev/hotmic
HOTMIC_DIR=$(mktemp -d) python3 - <<'PY'
import os, time, queue, io
import hotmic_whisper_worker as w
# Build a SessionManager directly and feed synthetic audio (no sox, no GPU).
d = os.environ["HOTMIC_DIR"]; os.makedirs(d + "/chunks", exist_ok=True)
typed = []
def fake_tx(p):
    import wave
    with wave.open(p,"rb") as wf: return f"chunk:{wf.getnframes()}"
sm = w.SessionManager(w.RingBuffer(200), d+"/chunks", fake_tx,
                      lambda t,win: typed.append(t), lambda: "win", print)
sm.start()
for _ in range(30): sm.ring.append(time.monotonic(), w.struct.pack(f"<{w.BLOCK_SAMPLES}h", *([10000]*w.BLOCK_SAMPLES)))
for _ in range(int(w.SILENCE_DUR/0.05)+1): sm.ring.append(time.monotonic(), b"\x00"*w.BLOCK_BYTES)
time.sleep(0.5); sm.stop()
print("TYPED:", typed)
assert typed, "no transcription produced"
print("OK")
PY
```
Expected: prints `TYPED: ['chunk:...']` then `OK`.

- [ ] **Step 8: Full test run + commit**

Run: `python3 -m pytest tests/test_worker.py -v`
Expected: all PASS

```bash
git add tests/test_worker.py hotmic_whisper_worker.py
git commit -m "feat: continuous capture + control FIFO + pause; rewire main(); idle 45min"
```

---

### Task 7: Scripts — signal the daemon instead of spawning sox

**Files:**
- Modify: `hotmic_start.sh:131-140`
- Modify: `hotmic_stop.sh`
- Create: `hotmic_pause.sh`

- [ ] **Step 1: `hotmic_start.sh` — replace the whisper-branch sox launch**

Replace lines 131-140 (the `sox ... -t raw "$DIR/audio.fifo" &` block through `log "Ready"` and `exit 0`) with:

```bash
    # The resident daemon owns continuous mic capture. Just tell it to start a
    # dictation session (it will include ~2s of pre-keypress audio via lookback).
    # The daemon holds control.fifo open O_RDWR, so this write does not block.
    if ! timeout 2 sh -c "printf 'START\n' > '$DIR/control.fifo'" 2>>"$LOG_FILE"; then
        log "WARN: control FIFO write (START) timed out"
    fi

    log "Dictation started (START -> daemon)"
    log "Ready"
    exit 0
fi
```

- [ ] **Step 2: `hotmic_stop.sh` — signal STOP instead of killing per-session sox**

After the existing `rm -f "$DIR/active"` line, add:

```bash
# Tell the whisper daemon to end the dictation session (it owns capture now;
# there is no per-session sox to kill).
timeout 2 sh -c "printf 'STOP\n' > '$DIR/control.fifo'" 2>/dev/null || true
```

(Leave the existing `kill_pid_file "$DIR/rec.pid"` — harmless no-op now — and the indicator/loop kills and `--daemon` handling unchanged.)

- [ ] **Step 3: Create `hotmic_pause.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
DIR="/tmp/hotmic"
CONTROL_FIFO="$DIR/control.fifo"
PAUSED_FLAG="$DIR/paused"

# Toggle: if currently paused, resume; otherwise pause. The daemon owns the
# PAUSED_FLAG; we read it only to decide direction.
if [ -f "$PAUSED_FLAG" ]; then
    cmd="RESUME"
else
    cmd="PAUSE"
fi
timeout 2 sh -c "printf '%s\n' '$cmd' > '$CONTROL_FIFO'" 2>/dev/null || true
echo "$cmd sent"
```

- [ ] **Step 4: Make executable + commit**

```bash
chmod +x hotmic_pause.sh
git add hotmic_start.sh hotmic_stop.sh hotmic_pause.sh
git commit -m "feat: scripts signal daemon (START/STOP/PAUSE) instead of spawning sox"
```

---

### Task 8: Deploy + manual verification

**Files:** none (deploy + live check)

- [ ] **Step 1: Deploy to ~/bin (kill python daemon by PID/comm, never `pkill -f` a pattern that matches this shell)**

```bash
cd ~/dev/hotmic
for f in hotmic_whisper_worker.py hotmic_start.sh hotmic_stop.sh hotmic_pause.sh; do cp "$f" ~/bin/"$f"; done
chmod +x ~/bin/hotmic_pause.sh
# stop old daemon: kill only python procs (comm filter), then clear stale state
for pid in $(pgrep -f hotmic_whisper_worker.py); do c=$(ps -o comm= -p "$pid"); [ "${c#python}" != "$c" ] && kill -9 "$pid"; done
rm -f /tmp/hotmic/whisper.ready /tmp/hotmic/audio.fifo /tmp/hotmic/control.fifo /tmp/hotmic/whisper_worker.pid /tmp/hotmic/paused
```

- [ ] **Step 2: Start the daemon via the normal path (so env/NVIDIA libs match)** — easiest is one real dictation trigger, or hand-spawn with the env block from start.sh lines 99-107. Then confirm:

```bash
sleep 8
grep -E "pre-warming model|model loaded|capture started" /tmp/hotmic/hotmic.log | tail -5
```
Expected: lines showing the daemon pre-warmed the model AND `capture started (mic armed)`.

- [ ] **Step 3: Manual dictation check (user)** — the critical acceptance test. Idle for ~1 minute, then dictate a sentence that begins immediately on keypress (e.g. "Lookback test one two three"). Confirm the **opening word is captured** (no mid-sentence start). Repeat after a longer idle (10+ min).

- [ ] **Step 4: Pause/resume check (user)** — run `~/bin/hotmic_pause.sh` (or the bound hotkey); confirm mic LED off and `paused (mic released)` in the log. Run again; confirm `resumed (mic armed)`.

- [ ] **Step 5: Commit nothing (deploy only). Record outcome in the conversation.** If the manual checks pass, the feature is done. If the opening word is still clipped, capture the log window and return to RCA — do not patch blindly.

---

## Notes for the implementer

- Keep `reader_loop` / `transcriber_loop` deletable only once `main()` no longer references them; remove them in Task 6 Step 6 to avoid dead code (the chunker + SessionManager replace them).
- `RESTART_IDLE_SEC` default is now `2700`; the live daemon keeps its old value until redeployed (Task 8).
- Deploy is manual `cp` to `~/bin` (live daemon runs from there). Four files now: worker + 3 scripts.
- The user should bind `hotmic_pause.sh` to a hotkey (their WM config, outside this repo).
