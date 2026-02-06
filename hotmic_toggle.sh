#!/usr/bin/env bash
set -euo pipefail

STATE_FILE="/tmp/hotmic/active"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -f "$STATE_FILE" ]; then
    "$SCRIPT_DIR/hotmic_stop.sh"
else
    "$SCRIPT_DIR/hotmic_start.sh"
fi
