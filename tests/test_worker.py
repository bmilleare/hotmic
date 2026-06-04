import sys, os, struct, io
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
