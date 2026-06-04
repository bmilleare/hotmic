#!/usr/bin/env bash
set -euo pipefail

# Toggle the whisper daemon's continuous mic capture. PAUSE fully releases the
# mic (LED off); RESUME re-arms it. The daemon owns the PAUSED_FLAG; we read it
# only to decide the toggle direction. Bind this to a hotkey.

DIR="/tmp/hotmic"
CONTROL_FIFO="$DIR/control.fifo"
PAUSED_FLAG="$DIR/paused"

if [ -f "$PAUSED_FLAG" ]; then
    cmd="RESUME"
else
    cmd="PAUSE"
fi

# The daemon holds control.fifo open O_RDWR, so this write does not block.
timeout 2 sh -c "printf '%s\n' '$cmd' > '$CONTROL_FIFO'" 2>/dev/null || true
echo "$cmd sent"
