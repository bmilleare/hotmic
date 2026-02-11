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

# Kill sox immediately so the mic stops.
# For whisper backend: this closes the FIFO write end, the daemon gets EOF,
# transcribes remaining audio, then waits for the next session.
# The daemon stays running with the model loaded — no restart needed.
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
    rm -f "$DIR/whisper.ready" "$DIR/audio.fifo"
fi

if [ -f "$DIR/hotmic.log" ]; then
    echo "[$(date '+%H:%M:%S')] Dictation stopped" >> "$DIR/hotmic.log"
fi
