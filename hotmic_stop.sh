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
# This closes the pipe → worker gets EOF → reader finishes.
# The worker continues running to load the model (if still loading)
# and transcribe remaining chunks, then exits on its own.
kill_pid_file "$DIR/rec.pid"

# Kill the indicator immediately (visual feedback that we stopped)
kill_pid_file "$DIR/indicator.pid"

# NOTE: We intentionally do NOT kill the whisper worker here.
# It needs time to finish model loading and transcribe the final audio.
# Stale workers are cleaned up by the start script when beginning a new session.

if [ -f "$DIR/hotmic.log" ]; then
    echo "[$(date '+%H:%M:%S')] Dictation stopped" >> "$DIR/hotmic.log"
fi
