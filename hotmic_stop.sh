#!/usr/bin/env bash
set -euo pipefail

DIR="/tmp/hotmic"

kill_pid_file() {
    local pf="$1"
    [ -f "$pf" ] || return 0
    local pid
    pid=$(cat "$pf" 2>/dev/null) || return 0
    kill "$pid" 2>/dev/null || true
    rm -f "$pf"
}

# Signal the loop to exit (LLM backend)
rm -f "$DIR/active"

# Whisper backend: tell the daemon to end the dictation session. The daemon owns
# continuous mic capture now, so there is no per-session sox to kill — it just
# stops transcribing and keeps the mic warm for the next session. The daemon
# holds control.fifo open O_RDWR, so this write does not block.
timeout 2 sh -c "printf 'STOP\n' > '$DIR/control.fifo'" 2>/dev/null || true

# Legacy LLM-backend per-chunk recorder (harmless no-op for whisper backend).
kill_pid_file "$DIR/rec.pid"

# Kill the indicator immediately (visual feedback that we stopped)
kill_pid_file "$DIR/indicator.pid"

# Kill the LLM backend loop if running
kill_pid_file "$DIR/loop.pid"

# NOTE: We intentionally do NOT kill the whisper daemon.
# It stays resident with the model loaded for instant next-session startup.
# To fully stop the daemon: hotmic_stop.sh --daemon

if [ "${1:-}" = "--daemon" ]; then
    kill_pid_file "$DIR/whisper_worker.pid"
    pkill -9 -f "hotmic_whisper_worker" 2>/dev/null || true
    rm -f "$DIR/whisper.ready" "$DIR/audio.fifo" "$DIR/control.fifo" "$DIR/paused"
fi

if [ -f "$DIR/hotmic.log" ]; then
    echo "[$(date '+%H:%M:%S')] Dictation stopped" >> "$DIR/hotmic.log"
fi
