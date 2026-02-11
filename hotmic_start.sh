#!/usr/bin/env bash
set -euo pipefail

# === Configuration ===
HOTMIC_BACKEND="${HOTMIC_BACKEND:-whisper}"  # "whisper" (local) or "llm" (OpenRouter)
OPENROUTER_MODEL="${OPENROUTER_MODEL:-google/gemini-2.0-flash-001}"
WHISPER_MODEL="${WHISPER_MODEL:-medium.en}"
WHISPER_DEVICE="${WHISPER_DEVICE:-cuda}"
SILENCE_THRESH="3%"          # voice-activity threshold (sox % for LLM, fraction for whisper)
SILENCE_DUR="0.8"            # seconds of silence to end a chunk
MAX_CHUNK_SEC="20"           # hard cap per chunk (ensures background transcription)
MIN_CHUNK_BYTES="2048"       # ignore chunks smaller than this (noise) — LLM backend only
CURL_TIMEOUT="15"            # API request timeout — LLM backend only
SOX_RATE="16000"
SOX_CHANNELS="1"
SOX_BITS="16"

DIR="/tmp/hotmic"
STATE_FILE="$DIR/active"
CHUNK_DIR="$DIR/chunks"
LOG_FILE="$DIR/hotmic.log"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

SYSTEM_PROMPT="You are a speech-to-text transcriber. Output ONLY the verbatim spoken words. Never add quotes, labels, timestamps, commentary, or formatting. If the audio contains only silence, noise, or is unintelligible, respond with exactly: [SILENCE]"

# === Quick dependency check (no heavy imports) ===
for cmd in sox xdotool python3; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "Required: $cmd" >&2; exit 1; }
done
if [ "$HOTMIC_BACKEND" = "llm" ]; then
    for cmd in curl jq; do
        command -v "$cmd" >/dev/null 2>&1 || { echo "Required: $cmd" >&2; exit 1; }
    done
fi

# === Load API key (only needed for LLM backend) ===
if [ "$HOTMIC_BACKEND" = "llm" ]; then
    if [ -z "${OPENROUTER_API_KEY:-}" ]; then
        for _envfile in "$SCRIPT_DIR/.env" "$HOME/.config/hotmic/env"; do
            if [ -f "$_envfile" ]; then
                # shellcheck source=/dev/null
                . "$_envfile"
                break
            fi
        done
        unset _envfile
    fi
    if [ -z "${OPENROUTER_API_KEY:-}" ]; then
        echo "OPENROUTER_API_KEY not set. See README.md for setup instructions." >&2
        exit 1
    fi
fi

# === Setup ===
mkdir -p "$DIR" "$CHUNK_DIR"
: > "$LOG_FILE"

log() { echo "[$(date '+%H:%M:%S')] $*" >> "$LOG_FILE"; }

# === Clean any previous session ===
"$SCRIPT_DIR/hotmic_stop.sh" --quiet 2>/dev/null || true

touch "$STATE_FILE"

# === Launch indicator immediately ===
python3 "$SCRIPT_DIR/hotmic_indicator.py" &
echo $! > "$DIR/indicator.pid"
log "Indicator PID $(cat "$DIR/indicator.pid")"

# ===========================================================================
# WHISPER BACKEND: continuous sox → pipe → Python worker (split + transcribe)
# ===========================================================================
if [ "$HOTMIC_BACKEND" = "whisper" ]; then
    # Kill any orphaned workers aggressively (SIGKILL) to ensure GPU memory is freed.
    # A worker killed mid-model-load can leave CUDA in a dirty state, causing the
    # next worker's model load to hang indefinitely.
    if pgrep -f "hotmic_whisper_worker" >/dev/null 2>&1; then
        pkill -9 -f "hotmic_whisper_worker" 2>/dev/null || true
        sleep 1  # allow GPU memory to be reclaimed by the kernel
    fi

    # Resolve NVIDIA library paths for CTranslate2
    NVIDIA_LIB_DIR="$(python3 -c 'import nvidia.cublas.lib; print(nvidia.cublas.lib.__path__[0])' 2>/dev/null || true)"
    CUDNN_LIB_DIR="$(python3 -c 'import nvidia.cudnn.lib; print(nvidia.cudnn.lib.__path__[0])' 2>/dev/null || true)"

    # Single continuous sox recording piped to the worker.
    # The worker handles splitting, transcription, and typing — no audio gaps.
    (
        # Disable pipefail so that sox being killed (non-zero exit) doesn't
        # abort the subshell before the worker can flush and transcribe.
        set +o pipefail

        sox -q -d -c "$SOX_CHANNELS" -r "$SOX_RATE" -b "$SOX_BITS" -e signed-integer -t raw - 2>>"$LOG_FILE" \
        | WHISPER_MODEL="$WHISPER_MODEL" WHISPER_DEVICE="$WHISPER_DEVICE" \
            MAX_CHUNK_SEC="$MAX_CHUNK_SEC" SILENCE_DUR="$SILENCE_DUR" SILENCE_THRESH="0.03" \
            LD_LIBRARY_PATH="${NVIDIA_LIB_DIR:+$NVIDIA_LIB_DIR:}${CUDNN_LIB_DIR:+$CUDNN_LIB_DIR:}${LD_LIBRARY_PATH:-}" \
            python3 "$SCRIPT_DIR/hotmic_whisper_worker.py" >> "$LOG_FILE" 2>&1 || true

        # When sox or worker exits, clean up
        if [ -f "$DIR/whisper_worker.pid" ]; then
            kill "$(cat "$DIR/whisper_worker.pid" 2>/dev/null)" 2>/dev/null || true
            rm -f "$DIR/whisper_worker.pid"
        fi
        pkill -f "hotmic_whisper_worker" 2>/dev/null || true
        rm -f "$DIR/whisper.ready"
        log "Recording pipeline exited"
        rm -f "$DIR/loop.pid"
    ) &
    LOOP_PID=$!
    echo "$LOOP_PID" > "$DIR/loop.pid"

    # Find the sox PID within the pipeline for the stop script
    sleep 0.2
    SOX_PID=$(pgrep -P "$LOOP_PID" -f "sox" 2>/dev/null | head -1 || true)
    [ -n "$SOX_PID" ] && echo "$SOX_PID" > "$DIR/rec.pid"

    # Also store the worker PID
    WORKER_PID=$(pgrep -P "$LOOP_PID" -f "hotmic_whisper_worker" 2>/dev/null | head -1 || true)
    [ -n "$WORKER_PID" ] && echo "$WORKER_PID" > "$DIR/whisper_worker.pid"

    log "Dictation started (pipeline PID $LOOP_PID)"
    log "Ready"
    exit 0
fi

# ===========================================================================
# LLM BACKEND: chunked sox recording with API transcription (original design)
# ===========================================================================

# === Transcribe via LLM (OpenRouter) ===
transcribe_chunk_llm() {
    local chunk_file="$1" b64_file="$2" chunk_num="$3"

    log "Chunk $chunk_num: $(stat -c%s "$chunk_file") bytes → OpenRouter ($OPENROUTER_MODEL)"

    base64 -w0 "$chunk_file" > "$b64_file"

    local response
    response=$(jq -nc \
        --arg model "$OPENROUTER_MODEL" \
        --arg system "$SYSTEM_PROMPT" \
        --rawfile audio "$b64_file" \
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
        rm -f "$chunk_file" "$b64_file"
        return
    }

    local text
    text=$(echo "$response" | jq -r '.choices[0].message.content // empty' 2>/dev/null || true)
    text="${text#"${text%%[![:space:]]*}"}"
    text="${text%"${text##*[![:space:]]}"}"
    case "$text" in
        "" | "[SILENCE]" | "..." | "Okay." | "Okay") text="" ;;
    esac

    if [ -n "$text" ]; then
        log "Transcribed: $text"
        xdotool type --clearmodifiers --delay 0 -- "$text "
    else
        local err
        err=$(echo "$response" | jq -r '.error.message // empty' 2>/dev/null || true)
        [ -n "$err" ] && log "API error: $err"
    fi

    rm -f "$chunk_file" "$b64_file"
}

(
    CHUNK_NUM=0
    FAIL_COUNT=0
    PENDING_CHUNK=""
    PENDING_B64=""
    PENDING_NUM=""
    log "Recording loop started (backend: llm, model: $OPENROUTER_MODEL)"

    while [ -f "$STATE_FILE" ]; do
        CHUNK_FILE="$CHUNK_DIR/chunk_${CHUNK_NUM}.wav"
        B64_FILE="$CHUNK_DIR/chunk_${CHUNK_NUM}.b64"
        CHUNK_NUM=$((CHUNK_NUM + 1))

        sox -q -d \
            -c "$SOX_CHANNELS" -r "$SOX_RATE" -b "$SOX_BITS" -e signed-integer \
            -t wav "$CHUNK_FILE" \
            silence 1 0.1 "$SILENCE_THRESH" 1 "$SILENCE_DUR" "$SILENCE_THRESH" \
            trim 0 "$MAX_CHUNK_SEC" 2>>"$LOG_FILE" &
        SOX_PID=$!
        echo "$SOX_PID" > "$DIR/rec.pid"

        if [ -n "$PENDING_CHUNK" ]; then
            transcribe_chunk_llm "$PENDING_CHUNK" "$PENDING_B64" "$PENDING_NUM"
            PENDING_CHUNK=""
        fi

        wait "$SOX_PID" 2>/dev/null
        SOX_EXIT=$?

        STOPPING=false
        [ -f "$STATE_FILE" ] || STOPPING=true

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

        FSIZE=$(stat -c%s "$CHUNK_FILE" 2>/dev/null || echo 0)
        if [ "$FSIZE" -lt "$MIN_CHUNK_BYTES" ]; then
            rm -f "$CHUNK_FILE"
            $STOPPING && break
            continue
        fi

        if $STOPPING; then
            transcribe_chunk_llm "$CHUNK_FILE" "$B64_FILE" "$((CHUNK_NUM - 1))"
            break
        fi

        PENDING_CHUNK="$CHUNK_FILE"
        PENDING_B64="$B64_FILE"
        PENDING_NUM="$((CHUNK_NUM - 1))"
    done

    if [ -n "$PENDING_CHUNK" ]; then
        transcribe_chunk_llm "$PENDING_CHUNK" "$PENDING_B64" "$PENDING_NUM"
    fi

    log "Recording loop exited"
    rm -f "$DIR/loop.pid"
) &
echo $! > "$DIR/loop.pid"

log "Dictation started (loop PID $(cat "$DIR/loop.pid"))"
log "Ready"
