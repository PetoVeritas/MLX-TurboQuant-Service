# Qwen3-TTS Dev User Test

This is the local-only test lane for Qwen3-TTS speech generation. It is meant to let a reviewer exercise the dev endpoint and listen to generated WAV files before any production wiring, push, or merge.

Safety boundary:

- Uses the dev repo only.
- Uses the isolated Qwen3-TTS runtime at `runtime/qwen3-tts-smoke/.venv/bin/python`.
- Uses the downloaded MLX model archive under `~/Documents/OpenClaw Assets/Downloads - LLM Models & OC Source Code/LLM Models/`.
- Starts a dedicated dev supervisor on `127.0.0.1:4037` by default.
- Writes only local temp files under `tmp/`.
- Does not touch the production LaunchAgent or production application-support service path.

## Commands

From the repo root:

```bash
scripts/qwen3-tts-dev-test start
scripts/qwen3-tts-dev-test status
scripts/qwen3-tts-dev-test speak "This is a local Qwen three T T S dev test."
scripts/qwen3-tts-dev-test speak --instruct "Speak warmly, with a calm but slightly amused delivery." "This is a local Qwen three T T S style test."
scripts/qwen3-tts-dev-test speak --ref-audio tmp/qwen3-tts-output/sample.wav --ref-text "Reference transcript text." "This is a local Qwen three T T S reference-audio test."
scripts/qwen3-tts-dev-test play "This opens the generated WAV in the default audio app."
scripts/qwen3-tts-dev-test stop
```

The `speak` command prints the JSON response from `POST /v1/audio/speech`, including:

- `audioPath`
- `durationSeconds`
- `fileSizeBytes`
- `metrics.totalMs`

The `play` command does the same render and then runs `open <audioPath>` so the generated WAV can be heard locally.

Use `--instruct` to pass Qwen3-TTS CustomVoice emotion/style instructions through to `mlx-audio --instruct`. This is the main control surface for delivery such as warm, amused, panicked, incredulous, calm, or excited.

Use `--ref-audio` with an optional `--ref-text` to pass a local WAV reference through to `mlx-audio --ref_audio` / `--ref_text`. The current dev API intentionally accepts local paths only; bytes/base64 upload can come later when artifact validation and retention are designed.

`speak` and `play` auto-start the dev server if it is not already running. Use `stop` when finished. If port `4037` is already occupied by another process, the helper exits instead of silently talking to the wrong service.

## Environment Overrides

```bash
QWEN3_TTS_PORT=4047 scripts/qwen3-tts-dev-test start
QWEN3_TTS_MODEL_PATH=/absolute/path/to/model scripts/qwen3-tts-dev-test speak "hello"
QWEN3_TTS_PYTHON=/absolute/path/to/.venv/bin/python scripts/qwen3-tts-dev-test speak "hello"
QWEN3_TTS_INSTRUCT="Speak in an incredulous tone." scripts/qwen3-tts-dev-test speak "Wait, that cannot be right."
QWEN3_TTS_REF_AUDIO=/absolute/path/to/reference.wav QWEN3_TTS_REF_TEXT="Reference transcript." scripts/qwen3-tts-dev-test speak "Reference-style output."
```

## Manual HTTP Shape

```bash
curl -sS -X POST http://127.0.0.1:4037/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-tts-local-1.7b-customvoice-4bit",
    "input": "This is a local Qwen three T T S dev test.",
    "speaker": "aiden",
    "instruct": "Speak warmly, with a calm but slightly amused delivery.",
    "referenceAudioPath": "/absolute/path/to/reference.wav",
    "referenceText": "Reference transcript text.",
    "format": "wav",
    "timeoutSeconds": 120
  }'
```

Known supported speakers from the downloaded model:

- `aiden`
- `dylan`
- `eric`
- `ono_anna`
- `ryan`
- `serena`
- `sohee`
- `uncle_fu`
- `vivian`
