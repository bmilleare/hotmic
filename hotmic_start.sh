#!/usr/bin/env bash
set -euo pipefail

# Source zshrc for API keys when launched from desktop keybindings
# (which don't run a zsh login shell)
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f "$HOME/.zshrc" ]; then
    OPENROUTER_API_KEY=$(grep -oP '^export OPENROUTER_API_KEY=\K.*' "$HOME/.zshrc" || true)
    export OPENROUTER_API_KEY
fi

# === Configuration ===
OPENROUTER_MODEL="${OPENROUTER_MODEL:-google/gemini-2.0-flash-001}"
SILENCE_THRESH="1%"       # voice-activity threshold
SILENCE_DUR="1.0"         # seconds of silence to end a chunk
MAX_CHUNK_SEC="30"        # hard cap per chunk
MIN_CHUNK_BYTES="2048"    # ignore chunks smaller than this (noise)
CURL_TIMEOUT="15"         # API request timeout (LLM inference is slower than dedicated STT)
SOX_RATE="16000"
SOX_CHANNELS="1"
SOX_BITS="16"

DIR="/tmp/hotmic"
STATE_FILE="$DIR/active"
CHUNK_DIR="$DIR/chunks"
LOG_FILE="$DIR/hotmic.log"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

SYSTEM_PROMPT="You are a speech-to-text transcriber. Output ONLY the verbatim spoken words. Never add quotes, labels, timestamps, commentary, or formatting. If the audio contains only silence, noise, or is unintelligible, respond with exactly: [SILENCE]"

# === Dependency check ===
for cmd in sox curl jq xdotool python3; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        exit 1
    fi
done

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    exit 1
fi

# === Clean any previous session ===
"$SCRIPT_DIR/hotmic_stop.sh" --quiet 2>/dev/null || true

# === Setup ===
mkdir -p "$DIR" "$CHUNK_DIR"
touch "$STATE_FILE"
: > "$LOG_FILE"

log() { echo "[$(date '+%H:%M:%S')] $*" >> "$LOG_FILE"; }

# === Launch pulsing indicator ===
python3 "$SCRIPT_DIR/hotmic_indicator.py" &
echo $! > "$DIR/indicator.pid"
log "Indicator PID $(cat "$DIR/indicator.pid")"

# === Recording loop (runs in background) ===
(
    CHUNK_NUM=0
    FAIL_COUNT=0
    log "Recording loop started (model: $OPENROUTER_MODEL)"

    while [ -f "$STATE_FILE" ]; do
        CHUNK_FILE="$CHUNK_DIR/chunk_${CHUNK_NUM}.wav"
        B64_FILE="$CHUNK_DIR/chunk_${CHUNK_NUM}.b64"
        CHUNK_NUM=$((CHUNK_NUM + 1))

        # Record one utterance: skip leading silence, stop after SILENCE_DUR of quiet
        sox -q -d \
            -c "$SOX_CHANNELS" -r "$SOX_RATE" -b "$SOX_BITS" -e signed-integer \
            -t wav "$CHUNK_FILE" \
            silence 1 0.1 "$SILENCE_THRESH" 1 "$SILENCE_DUR" "$SILENCE_THRESH" \
            trim 0 "$MAX_CHUNK_SEC" 2>>"$LOG_FILE" &
        SOX_PID=$!
        echo "$SOX_PID" > "$DIR/rec.pid"
        wait "$SOX_PID" 2>/dev/null
        SOX_EXIT=$?

        # Track whether we should exit after processing this chunk
        STOPPING=false
        [ -f "$STATE_FILE" ] || STOPPING=true

        # Handle sox failure (mic disconnected, etc.) — only if we're still running
        if [ "$SOX_EXIT" -ne 0 ] && ! $STOPPING; then
            FAIL_COUNT=$((FAIL_COUNT + 1))
            log "sox exit $SOX_EXIT (fail #$FAIL_COUNT)"
            if [ "$FAIL_COUNT" -ge 5 ]; then
                log "Too many sox failures — aborting"
                rm -f "$STATE_FILE"
                break
            fi
            sleep 0.5
            continue
        fi
        FAIL_COUNT=0

        # Skip tiny chunks (silence artifacts / no speech captured)
        FSIZE=$(stat -c%s "$CHUNK_FILE" 2>/dev/null || echo 0)
        if [ "$FSIZE" -lt "$MIN_CHUNK_BYTES" ]; then
            rm -f "$CHUNK_FILE"
            $STOPPING && break
            continue
        fi

        log "Chunk $((CHUNK_NUM - 1)): ${FSIZE} bytes → OpenRouter ($OPENROUTER_MODEL)"

        # === Base64-encode audio (use file to avoid shell arg-length limits) ===
        base64 -w0 "$CHUNK_FILE" > "$B64_FILE"

        # === Transcribe via OpenRouter chat completions ===
        RESPONSE=$(jq -nc \
            --arg model "$OPENROUTER_MODEL" \
            --arg system "$SYSTEM_PROMPT" \
            --rawfile audio "$B64_FILE" \
            '{
                model: $model,
                temperature: 0,
                max_tokens: 500,
                messages: [
                    {role: "system", content: $system},
                    {role: "user", content: [
                        {type: "text", text: "Transcribe:"},
                        {type: "input_audio", input_audio: {data: ($audio | rtrimstr("\n")), format: "wav"}}
                    ]}
                ]
            }' | curl -s --max-time "$CURL_TIMEOUT" \
            -H "Authorization: Bearer $OPENROUTER_API_KEY" \
            -H "Content-Type: application/json" \
            -d @- \
            "https://openrouter.ai/api/v1/chat/completions" 2>>"$LOG_FILE") || {
            log "curl failed"
            rm -f "$CHUNK_FILE" "$B64_FILE"
            continue
        }

        TEXT=$(echo "$RESPONSE" | jq -r '.choices[0].message.content // empty' 2>/dev/null || true)
        # Trim whitespace
        TEXT="${TEXT#"${TEXT%%[![:space:]]*}"}"
        TEXT="${TEXT%"${TEXT##*[![:space:]]}"}"
        # Skip silence/noise markers from the model
        case "$TEXT" in
            "" | "[SILENCE]" | "..." | "Okay." | "Okay") TEXT="" ;;
        esac

        if [ -n "$TEXT" ]; then
            log "Transcribed: $TEXT"
            xdotool type --clearmodifiers --delay 0 -- "$TEXT "
        else
            ERR=$(echo "$RESPONSE" | jq -r '.error.message // empty' 2>/dev/null || true)
            [ -n "$ERR" ] && log "API error: $ERR"
        fi

        rm -f "$CHUNK_FILE" "$B64_FILE"
        $STOPPING && break
    done

    log "Recording loop exited"
    rm -f "$DIR/loop.pid"
) &
echo $! > "$DIR/loop.pid"

log "Dictation started (loop PID $(cat "$DIR/loop.pid"))"
log "Ready"
