import sys, os, struct, io, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hotmic_whisper_worker as w

BB = w.BLOCK_BYTES


def _blk(byte=0):
    return bytes([byte]) * BB


def _speech_blk():
    # amplitude 10000 -> rms ~0.305, above SILENCE_THRESH (0.03)
    return struct.pack(f"<{w.BLOCK_SAMPLES}h", *([10000] * w.BLOCK_SAMPLES))


def _silence_blk():
    return b"\x00" * w.BLOCK_BYTES


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


# ---------------------------------------------------------------- RingBuffer

def test_ring_evicts_to_maxlen():
    ring = w.RingBuffer(maxlen=3)
    for i in range(5):
        ring.append(float(i), _blk(i % 256))
    snap = ring.snapshot()            # list[(ts, block)] oldest->newest
    assert [ts for ts, _ in snap] == [2.0, 3.0, 4.0]


def test_start_session_seeds_lookback_only():
    ring = w.RingBuffer(maxlen=200)
    for i in range(11):               # ts 0.0, 0.5, ..., 5.0
        ring.append(i * 0.5, _blk(i))
    q = ring.start_session(t_start=5.0, lookback_sec=2.0)   # keep ts >= 3.0
    blocks = _drain(q)
    assert len(blocks) == 5           # ts 3.0,3.5,4.0,4.5,5.0


def test_tee_after_start_no_gap_no_dup():
    ring = w.RingBuffer(maxlen=200)
    ring.append(0.0, _blk(1))         # pre-session, before lookback window
    ring.append(10.0, _blk(2))        # in lookback window
    q = ring.start_session(t_start=10.0, lookback_sec=2.0)
    ring.append(10.05, _blk(3))       # arrives after arm -> teed exactly once
    ring.append(10.10, _blk(4))
    blocks = _drain(q)
    assert blocks == [_blk(2), _blk(3), _blk(4)]


def test_stop_session_pushes_sentinel_and_disarms():
    ring = w.RingBuffer(maxlen=200)
    ring.append(1.0, _blk(1))
    q = ring.start_session(t_start=1.0, lookback_sec=2.0)
    ring.stop_session()
    ring.append(2.0, _blk(9))         # must NOT be teed after stop
    got = []
    while True:
        item = q.get(timeout=1)
        if item is None:
            break
        got.append(item)
    assert got == [_blk(1)]


# ----------------------------------------------------- split_blocks_to_chunks

def test_chunker_splits_on_silence():
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


# ----------------------------------------------------------------- capture_loop

def test_capture_loop_fills_ring_and_stops_on_eof():
    from threading import Event
    ring = w.RingBuffer(maxlen=200)
    n = 5
    src = io.BytesIO(b"".join(_silence_blk() for _ in range(n)))
    stop = Event()
    ticks = iter([float(i) for i in range(100)])
    w.capture_loop(src, ring, stop, clock=lambda: next(ticks))
    assert len(ring.snapshot()) == n     # stopped at EOF


class _ChunkedSource:
    """Simulates a pipe: read(n) returns at most `cap` bytes per call, b'' at EOF."""

    def __init__(self, data, cap):
        self.buf = data
        self.pos = 0
        self.cap = cap

    def read(self, n):
        n = min(n, self.cap)
        chunk = self.buf[self.pos:self.pos + n]
        self.pos += len(chunk)
        return chunk


def test_capture_loop_handles_short_reads():
    from threading import Event
    ring = w.RingBuffer(maxlen=200)
    n = 4
    src = _ChunkedSource(b"".join(_speech_blk() for _ in range(n)), cap=500)  # < 1600
    stop = Event()
    ticks = iter([float(i) for i in range(100)])
    w.capture_loop(src, ring, stop, clock=lambda: next(ticks))
    snap = ring.snapshot()
    assert len(snap) == n
    assert all(len(b) == w.BLOCK_BYTES for _, b in snap)


def test_capture_loop_tees_into_active_session():
    from threading import Event
    ring = w.RingBuffer(maxlen=200)
    ring.append(0.0, _silence_blk())
    q = ring.start_session(t_start=0.0, lookback_sec=2.0)
    src = io.BytesIO(b"".join(_speech_blk() for _ in range(3)))
    stop = Event()
    ticks = iter([0.01, 0.02, 0.03])
    w.capture_loop(src, ring, stop, clock=lambda: next(ticks))
    drained = _drain(q)
    assert len(drained) == 1 + 3         # seeded silence + 3 teed speech


# --------------------------------------------------------------- SessionManager

class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_session_manager_includes_lookback_and_types(tmp_path):
    typed = []

    def fake_transcribe(path):
        import wave
        with wave.open(path, "rb") as wf:
            return f"n={wf.getnframes()}"

    sm = w.SessionManager(
        ring=w.RingBuffer(maxlen=200),
        chunk_dir=str(tmp_path),
        transcribe_fn=fake_transcribe,
        type_fn=lambda text, wid: typed.append(text),
        get_window_fn=lambda: "win",
        clock=_FakeClock(),
        lookback_sec=2.0,
        trailing_sec=0,
    )
    sb = int(w.SILENCE_DUR / 0.05)
    # 2.0s of lookback already in the ring before start (ts 8.0..9.95)
    for i in range(40):
        sm.ring.append(8.0 + i * 0.05, _speech_blk())
    sm.clock.now = 10.0
    sm.start()                                # t_start=10.0 -> grabs the 40 blocks
    for i in range(20):                       # a bit more speech
        sm.ring.append(10.0 + i * 0.05, _speech_blk())
    for i in range(sb):                       # silence -> force a chunk
        sm.ring.append(11.0 + i * 0.05, _silence_blk())
    sm.stop()                                 # drains + joins
    assert typed, "expected at least one typed chunk"
    first_n = int(typed[0].split("=")[1])
    assert first_n >= 40 * w.BLOCK_SAMPLES    # first chunk contains the lookback


def test_stop_includes_trailing_audio(tmp_path):
    """Audio that arrives during the trailing window (after stop() is called) must
    still be captured — the tail of the final word."""
    import threading
    typed = []

    def fake_transcribe(path):
        import wave
        with wave.open(path, "rb") as wf:
            return f"n{wf.getnframes()}"       # must be a str (non-speech filter)

    sm = w.SessionManager(
        ring=w.RingBuffer(maxlen=400),
        chunk_dir=str(tmp_path),
        transcribe_fn=fake_transcribe,
        type_fn=lambda text, wid: typed.append(int(text[1:])),
        get_window_fn=lambda: None,
        trailing_sec=0.4,                      # real wall-clock trailing window
    )
    sm.start()
    for _ in range(20):                        # 20 blocks of speech during session
        sm.ring.append(time.monotonic(), _speech_blk())

    # A "final word" that lands 0.1s INTO the trailing window (after stop begins).
    def late_word():
        time.sleep(0.1)
        for _ in range(10):
            sm.ring.append(time.monotonic(), _speech_blk())
    threading.Thread(target=late_word).start()

    sm.stop()                                  # sleeps 0.4s; late blocks tee in
    total = sum(typed)
    assert total >= 30 * w.BLOCK_SAMPLES, (
        f"trailing audio dropped: got {total} samples, expected >= {30 * w.BLOCK_SAMPLES}")


# ------------------------------------------------------------------ control_loop

def test_control_thread_dispatches_commands(tmp_path):
    import time
    from threading import Event, Thread
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
        f.write("START\nSTOP\nPAUSE\nRESUME\n")
        f.flush()
    deadline = time.time() + 3
    while len(seen) < 4 and time.time() < deadline:
        time.sleep(0.02)
    stop.set()
    assert seen == ["START", "STOP", "PAUSE", "RESUME"]
