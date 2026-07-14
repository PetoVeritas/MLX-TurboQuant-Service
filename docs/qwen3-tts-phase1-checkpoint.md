# Qwen3-TTS Phase 1 Checkpoint

Status: ready for Sam review and Mauricio gate  
Branch: `qwen3-tts-backend-phase1`  
Checkpoint commit: `3b40aa8 docs: add qwen3 tts phase1 contract and smoke report`  

## What Changed

Tracked dev-repo changes:

- `docs/generative-backend-contract-phase1.md`
- `docs/qwen3-tts-phase1-smoke-report.md`

No service code changed in the Phase 1 checkpoint commit.

Important branch note:

- This branch was created from local `si-drone`, which was one commit ahead of `origin/main`.
- The inherited parent commit is `18c1fb9 fix: remove arbitrary 20k SI session context ceiling`.
- The Phase 1 checkpoint commit itself only adds the two docs files listed above.

## What Was Created Locally

Ignored runtime/output artifacts:

- `runtime/qwen3-tts-smoke/.venv`
- `tmp/qwen3-tts-output/qwen3-tts-phase1-smoke-aiden_000.wav`

Downloaded model:

```text
$HOME/Documents/OpenClaw Assets/Downloads - LLM Models & OC Source Code/LLM Models/mlx-community-Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit
```

The shared neutral runtime was not modified:

```text
$HOME/Library/Application Support/MLX-Shared-Runtime/mlx-python-runtime/.venv-mlxvlm-0347900
```

Production was not modified:

```text
$HOME/Library/Application Support/MLX-TurboQuant-Service/
```

## Smoke Result

Target model:

```text
mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit
```

Result:

- Model downloaded successfully.
- First attempt with speaker `Ethan` failed because unsupported.
- Second attempt with speaker `aiden` succeeded.
- Output: `tmp/qwen3-tts-output/qwen3-tts-phase1-smoke-aiden_000.wav`
- WAV metadata: 24 kHz, mono, PCM 16-bit, `6.080s`, `291,884` bytes.
- `mlx-audio` reported `2.79s` processing time, `2.18x` real-time factor, and `5.23GB` peak memory.
- `/usr/bin/time -l` reported `3.87s` wall time and `2,513,027,072` bytes max RSS.

## Phase 2 Recommendation

Proceed to Phase 2 only after Sam review and Mauricio approval.

Recommended Phase 2 scope:

- Build an isolated dev-only Qwen3-TTS lane/backend.
- Keep non-streaming audio generation first.
- Use `aiden` as the first known-supported default speaker.
- Expose supported speakers in `model_info()` / capability metadata.
- Return a local audio path first; defer bytes/base64 and streaming.
- Keep `:4019` E2B voice lane untouched.
- Keep `mlx-audio` runtime isolated until dependency compatibility is proven.

## Open Risks Before Phase 2

- Streaming/cancel lifecycle is not designed yet; keep it disabled initially.
- Artifact cleanup/retention policy needs implementation detail.
- Governor estimate should start conservative despite the promising 5.23GB smoke result.
- Supported speaker IDs are model-specific and must be discovered/validated at runtime.
- The parent `si-drone` commit needs to be accounted for when eventually merging/pushing.

## Sam Review

Sam reviewed the Phase 1 checkpoint and recommended approval.

Sam's non-blocking Phase 2 first tasks:

1. Define `speech.generate` error response shapes, including input-too-long, load-failed, generation-failed, and timeout.
2. Specify the structured return type for `estimate_memory(request)`, using the smoke's `5.23GB` MLX/audio peak as the initial baseline.
3. Define `model_info()` for Qwen3-TTS, including supported speakers, modes, sample rate, and model size. The failed `Ethan` speaker attempt is evidence this must be discoverable.

Sam also flagged:

- The branch base includes the unrelated but intentional SI Drone parent commit `18c1fb9`; acknowledge or rebase before Phase 2 code work or any eventual merge.
- Phase 2 should measure load time separately because the CLI did not report it in Phase 1.
- The checkpoint summary is docs-only and accurately reflects the reviewed work.

Review verdict:

```text
Approve Phase 1 checkpoint; address the three contract gaps first in Phase 2.
```

## Phase 2 Contract Checkpoint

Status: implemented locally, ready for review

This checkpoint addresses Sam's three first-task requirements without wiring HTTP speech generation yet.

Tracked code changes:

- `src/shared/backend_adapters.py`
- `src/worker/backends.py`
- `tests/test_backend_adapters.py`

Implemented contract surface:

- New backend adapter id: `mlx_audio_qwen3_tts`
- New worker backend contract class: `Qwen3TtsBackend`
- `model_info()` returns backend id, model id/path, model existence, model size, family, modes, sample rate, output format, output directory, default speaker, supported speakers, input limit, and streaming status.
- `estimate_memory(request)` returns a structured estimate using the Phase 1 `5.23GB` peak-memory smoke baseline, conservative estimated peak, recommended free-memory floor, request dimensions, confidence, basis, and component notes.
- `speech.generate` validation now has structured error shapes for:
  - `bad_request`
  - `input_too_long`
  - `unsupported_speaker`
  - `unsupported_format`
  - `load_failed`
  - `timeout`
  - `generation_failed`

Generation is not wired yet in this checkpoint. For a valid request with an existing model path, `speech_generate(...)` returns a structured `generation_failed` placeholder with `status: 501` and `phase: phase2_contract`.

Misrouted chat generation is rejected explicitly with `unsupported_request_family:Qwen3-TTS backend only supports speech.generate`.

Model speaker discovery:

- Reads `talker_config.spk_id` from the local model `config.json`, with `codec_config.spk_id` as a fallback for older/alternate artifacts.
- The downloaded model currently exposes `aiden`, `dylan`, `eric`, `ono_anna`, `ryan`, `serena`, `sohee`, `uncle_fu`, and `vivian`.
- `Ethan` is intentionally not advertised because it was not present in the model speaker map and failed the Phase 1 smoke.

Verification:

```text
PYTHONPATH=src python3 -m unittest tests.test_backend_adapters
PYTHONPATH=src python3 -m unittest tests.test_si_drone_sessions
PYTHONPATH=src python3 -m unittest tests.test_config tests.test_message_parts
```

All listed tests passed locally.

## Phase 2 Runtime/HTTP Checkpoint

Status: implemented locally, ready for review

This checkpoint wires the contract into a dev-only non-streaming speech generation path. It still does not push, deploy, or touch production.

Tracked code changes:

- `src/worker/backends.py`
- `src/worker/main.py`
- `src/supervisor/worker_manager.py`
- `src/supervisor/main.py`
- `tests/test_backend_adapters.py`

Implemented runtime surface:

- `Qwen3TtsBackend.speech_generate(request)` now runs `python -m mlx_audio.tts.generate` through the isolated Qwen3-TTS Python runtime.
- The backend preserves the venv `bin/python` symlink instead of resolving it to the Homebrew base interpreter; resolving the symlink drops the venv package context and cannot import `mlx_audio`.
- The backend writes WAV output to the configured repo-local output directory, discovers the generated file by request id prefix, reads WAV metadata, and returns local `audioPath`, `sampleRateHz`, `durationSeconds`, `fileSizeBytes`, and metrics.
- Worker command `speech_generate` dispatches to speech backends without routing through chat generation.
- Supervisor route `POST /v1/audio/speech` validates a non-streaming speech request, calls `WorkerManager.generate_speech(...)`, maps structured speech errors, and returns the local speech response shape.

Still deferred:

- Streaming audio.
- Cancellation/backpressure.
- Bytes/base64 response mode.
- Production artifact cleanup/retention policy.
- More precise load/Metal/RSS metrics.

Verification:

```text
PYTHONPATH=src python3 -m unittest tests.test_backend_adapters tests.test_si_drone_sessions tests.test_config tests.test_message_parts
python3 -m py_compile src/worker/backends.py src/worker/main.py src/supervisor/main.py src/supervisor/worker_manager.py
git diff --check
```

All listed checks passed locally.

Backend-level real render:

```text
audioPath: tmp/qwen3-tts-output/speech_2cb6a117aa99_000.wav
sampleRateHz: 24000
durationSeconds: 5.12
fileSizeBytes: 245804
totalMs: 2633
```

HTTP-level dev smoke:

```text
POST http://127.0.0.1:4037/v1/audio/speech
audioPath: tmp/qwen3-tts-output/speech_8b72d3b2cc25_000.wav
sampleRateHz: 24000
durationSeconds: 4.48
fileSizeBytes: 215084
totalMs: 1772
```

The temporary dev server on port `4037` was stopped after the smoke.

## Phase 2 Hardening Checkpoint

Status: implemented locally, ready for review

This checkpoint addresses the runtime/HTTP review notes while keeping the same dev-only, non-streaming scope.

Implemented hardening:

- Request `timeoutSeconds` is clamped by policy: requests above the backend's `maxTimeoutSeconds` now return structured `timeout_too_high` instead of letting the supervisor IPC timeout kill the worker first.
- `model_info()` now reports `defaultTimeoutSeconds`, `maxTimeoutSeconds`, and `retentionMaxFiles`.
- Qwen subprocess execution now uses a filtered allowlist environment and sets `PYTHONNOUSERSITE=1`, so the worker's broader environment is not blindly inherited.
- Failed subprocess details are reduced to bounded `stdoutTail` / `stderrTail` in worker-side error details.
- HTTP speech errors filter 5xx details before returning them to clients; raw subprocess stdout/stderr tails are not exposed through the HTTP response.
- The output directory has simple local retention by file count for `speech_*.wav`, preserving the current output and newest retained artifacts.

Still deferred:

- Age/size-based artifact retention.
- Production artifact policy.
- Packaging-safe replacement for the checkout-layout default path.
- Richer memory/load/Metal metrics.

## Packaging-Readiness Checkpoint

Status: implemented locally, ready for review

This checkpoint removes the Qwen3-TTS backend's dependency on deriving default runtime/output paths from the source file location. That pattern works in a source checkout but is brittle for wheel/app packaging.

Implemented packaging hardening:

- Added an explicit Qwen3-TTS `serviceRoot`.
- `serviceRoot` can be configured through `speech.serviceRoot`, `speech.service_root`, `paths.serviceRoot`, `paths.service_root`, or `MLX_GEMMA_SERVICE_ROOT`.
- If no explicit service root is configured, the backend uses the process working directory.
- Relative `speech.modelPath`, `speech.outputDir`, and `speech.pythonExecutable` now resolve against `serviceRoot`.
- Relative paths are normalized with `os.path.abspath(...)` without resolving symlinks, preserving the Qwen venv `bin/python` behavior.
- `model_info()` reports `serviceRoot` for diagnostics.
- The `mlx-audio` subprocess now runs with `cwd=serviceRoot`.
- The dev-user-test helper writes an explicit `speech.serviceRoot` into its generated temp config.

Verification:

- Added tests for config-driven service root path resolution.
- Added tests for `MLX_GEMMA_SERVICE_ROOT`.
- Focused test suite passed with 64 tests.
- `py_compile` passed for touched worker/supervisor modules.
- Real dev helper render still worked after the change:
  - `audioPath`: `tmp/qwen3-tts-output/speech_2b9704c4dde1_000.wav`
  - `sampleRateHz`: `24000`
  - `durationSeconds`: `6.8`
  - `fileSizeBytes`: `326444`
  - `metrics.totalMs`: `2263`

## Testing-Readiness Cleanup

See `docs/qwen3-tts-testing-readiness.md` for the current dev-test commands, verification snapshot, retained temp artifacts, and stragglers before any push/merge.
