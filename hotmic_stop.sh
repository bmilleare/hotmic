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

# Signal the loop to exit (it will finish any in-flight API call first)
rm -f "$DIR/active"

# Kill sox immediately so the mic stops
kill_pid_file "$DIR/rec.pid"

# Kill the indicator immediately (visual feedback that we stopped)
kill_pid_file "$DIR/indicator.pid"

# Do NOT kill the loop — let it finish the current transcription and exit naturally.
# It checks the state file after each API call and will stop on its own.

if [ -f "$DIR/hotmic.log" ]; then
    echo "[$(date '+%H:%M:%S')] Dictation stopped" >> "$DIR/hotmic.log"
fi
