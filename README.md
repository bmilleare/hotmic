# hotmic

Speech-to-text dictation for Linux. Speak into your microphone and text appears in whatever window has focus.

Supports two transcription backends:

- **Local Whisper** (default) — fast, private, no API key needed. Uses OpenAI's [Whisper](https://github.com/openai/whisper) running locally with GPU acceleration.
- **LLM via OpenRouter** — uses any audio-capable model on [OpenRouter](https://openrouter.ai/) (defaults to Gemini 2.0 Flash).

## How it works

1. `hotmic_toggle.sh` starts/stops dictation via a single keybinding
2. `sox` records from your microphone, splitting on speech pauses or every 10s
3. Each chunk is transcribed by the configured backend (local Whisper or OpenRouter LLM)
4. The result is typed into the focused window via `xdotool`
5. A pulsing red "REC" overlay badge shows while recording

Audio is chunked (max 10s each) to keep memory usage low and provide near-real-time transcription. The Whisper model stays loaded in a persistent worker process to avoid reload latency between chunks.

## Requirements

- Linux with X11
- `sox` (audio recording and silence detection)
- `xdotool`
- `python3` with PyGObject and cairo (for the recording indicator)

**For Whisper backend** (default):
- `python3` with `openai-whisper` installed (`pip install openai-whisper`)
- CUDA GPU recommended (falls back to CPU)

**For LLM backend:**
- `curl`
- `jq`
- An [OpenRouter API key](https://openrouter.ai/keys)

### Install dependencies (Debian/Ubuntu)

```bash
sudo apt install sox xdotool python3-gi python3-gi-cairo gir1.2-gtk-3.0

# For whisper backend:
pip install openai-whisper

# For LLM backend:
sudo apt install curl jq
```

### Install dependencies (Arch)

```bash
sudo pacman -S sox xdotool python-gobject python-cairo

# For whisper backend:
pip install openai-whisper

# For LLM backend:
sudo pacman -S curl jq
```

## Setup

#### 1. Clone this repo and make the scripts executable:

```bash
git clone https://github.com/bmilleare/hotmic.git
cd hotmic
chmod +x hotmic_toggle.sh hotmic_start.sh hotmic_stop.sh hotmic_indicator.py hotmic_whisper_worker.py
```

#### 2. Configure your backend

**Whisper (default)** — no configuration needed. Just ensure `openai-whisper` is installed.

**LLM** — set `HOTMIC_BACKEND=llm` and provide your OpenRouter API key:

```bash
echo 'OPENROUTER_API_KEY="sk-or-v1-your-key-here"' > /path/to/hotmic/.env
```

Alternatively, create `~/.config/hotmic/env` with the same content, or export it in your shell profile. The script checks these locations in order:

1. Environment variable (already set)
2. `.env` file next to the script
3. `~/.config/hotmic/env`

#### 3. Bind `hotmic_toggle.sh` to a keyboard shortcut in your desktop environment's settings. For example, in GNOME:

```
Settings > Keyboard > Custom Shortcuts > Add:
  Name: Dictation
  Command: /path/to/hotmic/hotmic_toggle.sh
  Shortcut: (your choice, e.g. Insert)
```

#### 4. Press your shortcut, speak, press it again. Text appears in the focused window.

## Configuration

Edit the variables at the top of `hotmic_start.sh`, or override them via environment variables:

| Variable | Default | Description |
|---|---|---|
| `HOTMIC_BACKEND` | `whisper` | `whisper` (local) or `llm` (OpenRouter) |
| `WHISPER_MODEL` | `tiny` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |
| `WHISPER_DEVICE` | `cuda` | `cuda` for GPU, `cpu` for CPU-only |
| `OPENROUTER_MODEL` | `google/gemini-2.0-flash-001` | Any audio-capable model on OpenRouter (LLM backend only) |
| `SILENCE_START_THRESH` | `3%` | Threshold to detect speech start (must be above ambient noise) |
| `SILENCE_STOP_THRESH` | `3%` | Threshold to detect pauses (must be above ambient noise) |
| `SILENCE_DUR` | `0.8` | Seconds of silence before a chunk ends |
| `MAX_CHUNK_SEC` | `10` | Hard cap per chunk (shorter = more responsive) |

## Files

| File | Purpose |
|---|---|
| `hotmic_toggle.sh` | Start/stop dictation (bind to a hotkey) |
| `hotmic_start.sh` | Main recording + transcription loop |
| `hotmic_stop.sh` | Stops recording and cleans up |
| `hotmic_indicator.py` | Pulsing "REC" overlay badge |
| `hotmic_whisper_worker.py` | Persistent Whisper process (loads model once) |

## Logs

Logs are written to `/tmp/hotmic/hotmic.log` for debugging.

## License

MIT
