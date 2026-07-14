# Qwen3-TTS Testing Readiness

Status: MLX service development wrapped for handoff/swap coordination.

This document captures the state after the Qwen3-TTS backend contract, runtime wiring, hardening, dev-user-test helper, packaging-readiness cleanup, style instruction support, and reference-audio controls checkpoints. It is intentionally scoped to the dev branch and does not authorize production changes or a GitHub push.

## Current Branch

```text
branch: qwen3-tts-backend-phase1
head: fe0585a feat: add qwen3 tts reference audio controls
```

The branch contains these Qwen3-TTS commits above the inherited SI Drone base:

```text
<current head> fe0585a feat: add qwen3 tts reference audio controls
fd62fa8 feat: expose qwen3 tts style instruction helper
1e8ca4b docs: record qwen3 tts testing readiness review
fe8c4ef docs: add qwen3 tts testing readiness checklist
4e8f87d fix: make qwen3 tts paths packaging safe
784bc57 docs: add qwen3 tts dev user test helper
697443f fix: harden qwen3 tts speech runtime
add22da feat: wire qwen3 tts speech generation route
0e05898 feat: add qwen3 tts speech backend contract
b6156f9 docs: record qwen3 tts phase1 review
68193bf docs: add qwen3 tts phase1 checkpoint summary
3b40aa8 docs: add qwen3 tts phase1 contract and smoke report
```

Important branch-base note:

- This branch still includes inherited commit `18c1fb9 fix: remove arbitrary 20k SI session context ceiling` below the Qwen commits.
- Before push or merge, either intentionally rebase onto the intended base or explicitly acknowledge that SI Drone parent commit as part of the branch history.

## Ready-To-Test Surface

Local dev speech endpoint:

```text
POST http://127.0.0.1:4037/v1/audio/speech
```

Preferred helper commands from the repo root:

```bash
scripts/qwen3-tts-dev-test play "This is a local Qwen three T T S dev test."
scripts/qwen3-tts-dev-test stop
```

For command-line inspection without opening the WAV:

```bash
scripts/qwen3-tts-dev-test speak "This is a local Qwen three T T S dev test."
```

The helper:

- writes a generated config, PID, and log under `tmp/`
- starts a dedicated dev supervisor on `127.0.0.1:4037`
- refuses to start if that port is already occupied
- writes `speech.serviceRoot` explicitly
- uses the isolated Qwen runtime at `runtime/qwen3-tts-smoke/.venv/bin/python`
- writes generated WAV files under `tmp/qwen3-tts-output/`
- does not touch production
- supports `--instruct` for style/instruction experiments
- supports `--ref-audio` and `--ref-text` for local reference-audio experiments

## Verification Snapshot

Latest full local verification on 2026-07-02 after `fe0585a`:

```text
PYTHONPATH=src python3 -m unittest discover -s tests
Ran 75 tests in 0.022s
OK

bash -n scripts/qwen3-tts-dev-test
OK

python3 -m compileall -q src scripts/qwen3-tts-dev-test
OK

git diff --check
OK

scripts/qwen3-tts-dev-test status
Qwen3-TTS dev server not running
```

Latest real Qwen render:

```text
audioPath: tmp/qwen3-tts-output/speech_9188e7b505d0_000.wav
format: wav
sampleRateHz: 24000
channels: 1
durationSeconds: 7.68
fileSizeBytes: 368684
metrics.totalMs: 2688
```

Sam review status:

- `0e05898`: approved after speaker-discovery blocker was fixed.
- `add22da`: approved, no blockers.
- `697443f`: approved, no blockers.
- `784bc57`: approved, no blockers.
- `4e8f87d`: approved, no blockers.
- Testing-readiness checklist checkpoint: approved, no blockers.
- `fd62fa8`: style instruction helper checkpoint reviewed.
- `fe0585a`: reference audio controls checkpoint approved, no blockers.

Sam's non-blocking notes for `fe0585a`:

- Supervisor does not type-check reference fields before worker validation; backend catches them.
- Reference audio path validation checks existence, not audio file type.
- CLI uses hyphenated helper flags while API fields use camelCase; keep naming conventions clear for future flags.

## Local Temp Artifacts

Current Qwen temp output footprint:

```text
tmp/qwen3-tts-output: about 5.5 MB, 15 WAV files
tmp total: about 13 MB
```

These files are retained as dev evidence for now. The backend already has count-based retention for generated `speech_*.wav` files, and the helper keeps outputs under `tmp/`.

Do not clean these before Mauricio's first listening pass unless explicitly asked.

## Stragglers Before Push/Merge

Required before push/merge:

- Resolve branch-base history for inherited `18c1fb9`.
- Decide whether this remains a dev-only branch or becomes the release branch.
- Confirm no production LaunchAgent/config/service paths are changed.
- Coordinate with Sam/Mauricio so no one is using the live MLX service during any production swap.
- Re-run the verification snapshot after any final merge/rebase.

Still intentionally deferred from this batch:

- Streaming audio.
- Cancellation/backpressure for speech generation.
- Bytes/base64 audio response mode.
- Production artifact retention policy beyond local count-based dev cleanup.
- Richer load/RSS/Metal metrics.
- Additional speaker/style UX beyond explicit supported speaker ids.
- Production integration with the outbound OpenClaw voice wrapper.
- Wrapper-level pronunciation overrides, default Tyler reference selection, speed defaults, short Signal filenames, and post-render silence/end-runway handling.

Out of scope for this batch:

- Replacing the current Signal/Telegram voice delivery stack.
- Real-time voice-to-voice.
- Production deployment without an explicit migration window.
- GitHub push.

## Production Swap Gate

Do not move this branch into production until Mauricio coordinates a quiet window with Sam and confirms the live MLX service is not in use.

Service-side development is wrapped at `fe0585a` for the current scope:

- Qwen3-TTS speech backend exists and rejects misrouted chat requests.
- Dev helper can start/status/stop/speak/play against an isolated local supervisor.
- Runtime/model/output paths are explicit and packaging safer than source-relative defaults.
- Subprocess execution is hardened with a filtered environment.
- Style instruction forwarding is wired through to `mlx-audio --instruct`.
- Reference audio/reference text forwarding is wired through to `mlx-audio --ref_audio` / `--ref_text`.
- Unit tests, shell syntax, compileall, and diff whitespace checks pass.
- Dev server is currently not running.
