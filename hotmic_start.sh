#!/usr/bin/env bash
set -euo pipefail

# === Configuration ===
HOTMIC_BACKEND="${HOTMIC_BACKEND:-whisper}"  # "whisper" (local) or "llm" (OpenRouter)
OPENROUTER_MODEL="${OPENROUTER_MODEL:-google/gemini-2.0-flash-001}"
WHISPER_MODEL="${WHISPER_MODEL:-medium.en}"
WHISPER_DEVICE="${WHISPER_DEVICE:-cuda}"
SILENCE_START_THRESH="3%"    # threshold to detect speech start (must be above ambient noise)
SILENCE_STOP_THRESH="3%"     # threshold to detect pause (must be above ambient noise)
SILENCE_DUR="0.8"            # seconds of silence to end a chunk
MAX_CHUNK_SEC="10"           # hard cap per chunk (ensures background transcription)
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

# === Setup (create dirs first so cleanup + indicator can use them) ===
mkdir -p "$DIR" "$CHUNK_DIR"
: > "$LOG_FILE"

log() { echo "[$(date '+%H:%M:%S')] $*" >> "$LOG_FILE"; }

# === Clean any previous session (non-blocking) ===
"$SCRIPT_DIR/hotmic_stop.sh" --quiet 2>/dev/null || true

touch "$STATE_FILE"

# === Launch indicator + recording immediately (no delay) ===
python3 "$SCRIPT_DIR/hotmic_indicator.py" &
echo $! > "$DIR/indicator.pid"
log "Indicator PID $(cat "$DIR/indicator.pid")"

# === Launch whisper worker (model loads in background) ===
WHISPER_FIFO="$DIR/whisper.fifo"
if [ "$HOTMIC_BACKEND" = "whisper" ]; then
    # Kill any orphaned workers from previous sessions before starting fresh
    pkill -f "hotmic_whisper_worker" 2>/dev/null || true
    rm -f "$WHISPER_FIFO" "$DIR/whisper.ready"
    mkfifo "$WHISPER_FIFO"
    # Add pip-installed NVIDIA library paths for CTranslate2
    NVIDIA_LIB_DIR="$(python3 -c 'import nvidia.cublas.lib; print(nvidia.cublas.lib.__path__[0])' 2>/dev/null || true)"
    CUDNN_LIB_DIR="$(python3 -c 'import nvidia.cudnn.lib; print(nvidia.cudnn.lib.__path__[0])' 2>/dev/null || true)"
    WHISPER_MODEL="$WHISPER_MODEL" WHISPER_DEVICE="$WHISPER_DEVICE" \
        LD_LIBRARY_PATH="${NVIDIA_LIB_DIR:+$NVIDIA_LIB_DIR:}${CUDNN_LIB_DIR:+$CUDNN_LIB_DIR:}${LD_LIBRARY_PATH:-}" \
        python3 "$SCRIPT_DIR/hotmic_whisper_worker.py" >> "$LOG_FILE" 2>&1 &
    echo $! > "$DIR/whisper_worker.pid"
    log "Whisper worker PID $(cat "$DIR/whisper_worker.pid")"
fi

# === Transcribe via local whisper (send to persistent worker) ===
transcribe_chunk_whisper() {
    local chunk_file="$1" chunk_num="$2"
    local txt_file="${chunk_file%.wav}.txt"

    log "Chunk $chunk_num: $(stat -c%s "$chunk_file") bytes → whisper worker"

    # Wait for worker to be ready (model loading can take 10-15s on first run)
    local waited=0
    while [ ! -f "$DIR/whisper.ready" ] && [ "$waited" -lt 150 ]; do
        sleep 0.2
        waited=$((waited + 1))
    done
    if [ ! -f "$DIR/whisper.ready" ]; then
        log "Whisper worker not ready after 30s, skipping chunk $chunk_num"
        rm -f "$chunk_file"
        return
    fi

    # Send chunk path to worker (non-blocking with timeout)
    timeout 2 bash -c "echo '$chunk_file' > '$WHISPER_FIFO'" 2>/dev/null || {
        log "FIFO write timed out, skipping chunk $chunk_num"
        rm -f "$chunk_file"
        return
    }

    # Wait for result (worker writes .txt and deletes .wav)
    waited=0
    while [ -f "$chunk_file" ] && [ "$waited" -lt 30 ]; do
        sleep 0.2
        waited=$((waited + 1))
    done

    if [ -f "$txt_file" ]; then
        local text
        text=$(cat "$txt_file")
        rm -f "$txt_file"
        if [ -n "$text" ]; then
            log "Transcribed: $text"
            xdotool type --clearmodifiers --delay 0 -- "$text "
        fi
    fi
}

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

# === Dispatch to the configured backend ===
transcribe_chunk() {
    if [ "$HOTMIC_BACKEND" = "whisper" ]; then
        transcribe_chunk_whisper "$1" "$3"
    else
        transcribe_chunk_llm "$1" "$2" "$3"
    fi
}

# === Recording loop (runs in background) ===
(
    CHUNK_NUM=0
    FAIL_COUNT=0
    PENDING_CHUNK=""
    PENDING_B64=""
    PENDING_NUM=""
    if [ "$HOTMIC_BACKEND" = "whisper" ]; then
        log "Recording loop started (backend: whisper, model: $WHISPER_MODEL)"
    else
        log "Recording loop started (backend: llm, model: $OPENROUTER_MODEL)"
    fi

    while [ -f "$STATE_FILE" ]; do
        CHUNK_FILE="$CHUNK_DIR/chunk_${CHUNK_NUM}.wav"
        B64_FILE="$CHUNK_DIR/chunk_${CHUNK_NUM}.b64"
        CHUNK_NUM=$((CHUNK_NUM + 1))

        # Record one utterance: skip leading silence, stop after SILENCE_DUR of quiet
        sox -q -d \
            -c "$SOX_CHANNELS" -r "$SOX_RATE" -b "$SOX_BITS" -e signed-integer \
            -t wav "$CHUNK_FILE" \
            silence 1 0.1 "$SILENCE_START_THRESH" 1 "$SILENCE_DUR" "$SILENCE_STOP_THRESH" \
            trim 0 "$MAX_CHUNK_SEC" 2>>"$LOG_FILE" &
        SOX_PID=$!
        echo "$SOX_PID" > "$DIR/rec.pid"

        # While sox records the next chunk, transcribe the previous one
        if [ -n "$PENDING_CHUNK" ]; then
            transcribe_chunk "$PENDING_CHUNK" "$PENDING_B64" "$PENDING_NUM"
            PENDING_CHUNK=""
        fi

        wait "$SOX_PID" 2>/dev/null
        SOX_EXIT=$?

        STOPPING=false
        [ -f "$STATE_FILE" ] || STOPPING=true

        FSIZE_DBG=$(stat -c%s "$CHUNK_FILE" 2>/dev/null || echo 0)
        log "sox finished: exit=$SOX_EXIT size=$FSIZE_DBG stopping=$STOPPING chunk=$CHUNK_FILE"

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
            log "Chunk too small ($FSIZE bytes < $MIN_CHUNK_BYTES), skipping"
            rm -f "$CHUNK_FILE"
            $STOPPING && break
            continue
        fi

        # If stopping, process this final chunk immediately and exit
        if $STOPPING; then
            transcribe_chunk "$CHUNK_FILE" "$B64_FILE" "$((CHUNK_NUM - 1))"
            break
        fi

        # Queue this chunk for processing during the next recording
        PENDING_CHUNK="$CHUNK_FILE"
        PENDING_B64="$B64_FILE"
        PENDING_NUM="$((CHUNK_NUM - 1))"
    done

    # Process any remaining pending chunk
    if [ -n "$PENDING_CHUNK" ]; then
        transcribe_chunk "$PENDING_CHUNK" "$PENDING_B64" "$PENDING_NUM"
    fi

    # Clean up whisper worker now that all transcription is done
    if [ -f "$DIR/whisper_worker.pid" ]; then
        kill "$(cat "$DIR/whisper_worker.pid" 2>/dev/null)" 2>/dev/null || true
        rm -f "$DIR/whisper_worker.pid"
    fi
    rm -f "$DIR/whisper.fifo" "$DIR/whisper.ready"
    pkill -f "hotmic_whisper_worker" 2>/dev/null || true

    log "Recording loop exited"
    rm -f "$DIR/loop.pid"
) &
echo $! > "$DIR/loop.pid"

log "Dictation started (loop PID $(cat "$DIR/loop.pid"))"
log "Ready"
