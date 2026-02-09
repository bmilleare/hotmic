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

# Signal the loop to exit
rm -f "$DIR/active"

# Kill sox immediately so the mic stops.
# For whisper backend: this closes the pipe, which triggers the worker to
# flush remaining audio, transcribe it, and exit cleanly.
kill_pid_file "$DIR/rec.pid"

# Kill the indicator immediately (visual feedback that we stopped)
kill_pid_file "$DIR/indicator.pid"

# Safety net: kill any orphaned whisper workers after giving the pipeline
# time to flush (background, with delay)
(
    sleep 5
    pkill -f "hotmic_whisper_worker" 2>/dev/null || true
    rm -f "$DIR/whisper_worker.pid" "$DIR/whisper.ready"
) &

if [ -f "$DIR/hotmic.log" ]; then
    echo "[$(date '+%H:%M:%S')] Dictation stopped" >> "$DIR/hotmic.log"
fi
