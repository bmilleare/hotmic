# hotmic

Speech-to-text dictation for Linux. Speak into your microphone and text appears in whatever window has focus.

Uses [OpenRouter](https://openrouter.ai/) to transcribe audio via any audio-capable model (defaults to Gemini 2.0 Flash). Audio is recorded in chunks using silence detection, sent to the API, and the transcribed text is typed into the active window.

## How it works

1. `hotmic_toggle.sh` starts/stops dictation via a single keybinding
2. `sox` records from your microphone, splitting on natural speech pauses (1s of silence)
3. Each chunk is base64-encoded and sent to OpenRouter for transcription
4. The result is typed into the focused window via `xdotool`
5. A pulsing red "REC" overlay badge shows while recording

## Requirements

- Linux with X11
- `sox` (audio recording and silence detection)
- `curl`
- `jq`
- `xdotool`
- `python3` with PyGObject and cairo (for the recording indicator)
- An [OpenRouter API key](https://openrouter.ai/keys)

### Install dependencies (Debian/Ubuntu)

```bash
sudo apt install sox curl jq xdotool python3-gi python3-gi-cairo gir1.2-gtk-3.0
```

### Install dependencies (Arch)

```bash
sudo pacman -S sox curl jq xdotool python-gobject python-cairo
```

## Setup

#### 1. Clone this repo and make the scripts executable:

```bash
git clone https://github.com/bmilleare/hotmic.git
cd hotmic
chmod +x hotmic_toggle.sh hotmic_start.sh hotmic_stop.sh hotmic_indicator.py
```

#### 2. Set your OpenRouter API key

Create a `.env` file next to the scripts (recommended — works with any shell and keyboard shortcuts):

```bash
echo 'OPENROUTER_API_KEY="sk-or-v1-your-key-here"' > /path/to/hotmic/.env
```

Alternatively, create `~/.config/hotmic/env` with the same content, or export it in your shell profile (`~/.bashrc`, `~/.zshrc`, etc.). The script checks these locations in order:

1. Environment variable (already set)
2. `.env` file next to the script
3. `~/.config/hotmic/env`

#### 3. Bind `hotmic_toggle.sh` to a keyboard shortcut in your desktop environment's settings. For example, in GNOME:

```
Settings > Keyboard > Custom Shortcuts > Add:
  Name: Dictation
  Command: /path/to/hotmic/hotmic_toggle.sh
  Shortcut: (your choice, e.g. Super+D)
```

#### 4. Press your shortcut, speak, press it again. Text appears in the focused window.

## Configuration

Edit the variables at the top of `hotmic_start.sh`:

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_MODEL` | `google/gemini-2.0-flash-001` | Any audio-capable model on OpenRouter |
| `SILENCE_THRESH` | `1%` | Voice activity detection threshold |
| `SILENCE_DUR` | `1.0` | Seconds of silence before a chunk ends |
| `MAX_CHUNK_SEC` | `30` | Maximum duration per chunk |

You can also override the model via environment variable:

```bash
export OPENROUTER_MODEL="google/gemini-2.5-flash-lite"
```

## Logs

Logs are written to `/tmp/hotmic/hotmic.log` for debugging.

## License

MIT
