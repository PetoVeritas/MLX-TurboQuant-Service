# Generative Backend Contract Phase 1

Status: Phase 1 development spec  
Branch: `qwen3-tts-backend-phase1`  
Baseline model: `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit`  

## Purpose

Define the backend contract needed to add Qwen3-TTS as the first speech-generation backend without destabilizing the existing text, multimodal, DiffusionGemma, or SI Drone paths.

This is not a production wiring plan. Phase 1 produces a contract, isolation plan, and smoke evidence. Production remains untouched.

## Existing Architecture Anchors

Relevant files:

| Area | File |
|---|---|
| Backend ids and runtime modality descriptors | `src/shared/backend_adapters.py` |
| Typed input parts and modality validation | `src/shared/parts.py` |
| Worker backend result and stream dataclasses | `src/worker/backends.py` |
| Supervisor HTTP routes | `src/supervisor/main.py` |
| Worker lifecycle and request dispatch | `src/supervisor/worker_manager.py` |
| Config loading and env overrides | `src/shared/config.py` |
| Default and example configs | `config/default.json`, `config/*.example.json` |
| Existing backend adapter tests | `tests/test_backend_adapters.py` |
| Existing typed-part tests | `tests/test_message_parts.py` |

Current backend ids:

- `stub`
- `mlx_vlm_turboquant`
- `mlx_vlm_diffusion_gemma`

Proposed new backend id:

- `mlx_audio_qwen3_tts`

## Contract Goals

The contract should separate shared lifecycle behavior from model-family-specific request and output types.

Shared lifecycle:

```text
load_model(config)
unload_model(reason)
health()
model_info()
capabilities()
estimate_memory(request?)
generate(request)
stream(request)
cancel(job_id)
```

The service should not pretend all backends are chat backends. The lifecycle can be shared, but request families must stay explicit.

## Request Families

Initial request family map:

| Family | Purpose | External route |
|---|---|---|
| `chat.generate` | non-streaming chat/text/multimodal response | `/v1/chat/completions` |
| `chat.stream` | SSE text/chat streaming | `/v1/chat/completions` with `stream=true` |
| `speech.generate` | non-streaming text-to-speech output | proposed `/v1/audio/speech` |
| `speech.stream` | streaming audio output | proposed `/v1/audio/speech` with stream mode, later |
| `speech.clone` | speech generation with reference audio/transcript | internal shape under `speech.generate` |
| `speech.design` | speech generation with style/voice instruction | internal shape under `speech.generate` |

Do not route Qwen3-TTS through `/v1/chat/completions`. It is speech generation, not assistant text generation.

## Capability Shape

Backends should expose configured, backend-supported, and effective capability metadata.

Example for Qwen3-TTS:

```json
{
  "backendId": "mlx_audio_qwen3_tts",
  "family": "tts",
  "modalities": {
    "input": ["text", "audio_ref", "reference_transcript", "speaker_id", "style_instruction"],
    "output": ["audio"]
  },
  "features": {
    "streaming": false,
    "cancel": false,
    "voiceClone": true,
    "voiceDesign": true,
    "customVoice": true,
    "toolCalls": false
  },
  "limits": {
    "maxTextChars": null,
    "maxReferenceAudioSeconds": null,
    "maxConcurrentRequests": 1
  }
}
```

Phase 2 can mark streaming/cancel as false until the binary stream lifecycle is designed and tested.

## Status Surface Requirements

Extend the existing status idea, not necessarily the implementation in Phase 1.

`/ready` should eventually include:

- active backend id
- backend family
- worker state
- configured capabilities
- backend-supported capabilities
- effective capabilities
- whether speech generation is enabled

`/v1/models` should eventually include:

- model id
- backend id
- backend family
- supported request families
- modality input/output metadata
- generated-output formats

`/admin/stats` should eventually include:

- backend id/family
- model load state
- last request family
- memory estimate values
- loaded memory metrics when available
- generated audio counts/durations for speech backends

## Qwen3-TTS Request Contract

Initial non-streaming speech request:

```json
{
  "model": "qwen3-tts-local-1.7b-customvoice-4bit",
  "input": "Plain spoken text to synthesize.",
  "voice": {
    "mode": "custom_voice",
    "speaker": null,
    "styleInstruction": null,
    "referenceAudioPath": null,
    "referenceTranscript": null
  },
  "format": "wav",
  "sampleRateHz": null,
  "timeoutSeconds": 120
}
```

Notes:

- `input` is already speech-ready text. The voice-gateway layer owns sentence coalescing and risky-word protection.
- `voice.mode` should support `custom_voice`, `voice_design`, and later `voice_clone`.
- File-path inputs must be local-only and validated. Do not accept arbitrary remote URLs in the service.
- Phase 2 should start with one simple mode before reference audio complexity.

## Qwen3-TTS Response Contract

Preferred Phase 2 non-streaming response:

```json
{
  "id": "speech_...",
  "object": "audio.speech",
  "model": "qwen3-tts-local-1.7b-customvoice-4bit",
  "backend": "mlx_audio_qwen3_tts",
  "format": "wav",
  "sampleRateHz": 24000,
  "durationSeconds": 3.42,
  "audioPath": "/absolute/local/path.wav",
  "metrics": {
    "loadMs": 1234,
    "firstAudioMs": null,
    "totalMs": 4567,
    "rssGb": null,
    "metalActiveGb": null,
    "metalPeakGb": null,
    "generatedAudioSeconds": 3.42
  }
}
```

Return a local path first. Bytes/base64 and streaming can come later after artifact cleanup and backpressure behavior are settled.

## Artifact Policy

Generated speech should not land in the workspace root.

Recommended Phase 2 dev output root:

```text
~/Documents/OpenClaw Assets/Apps/MLX-TurboQuant-Service-dev/tmp/qwen3-tts-output/
```

The backend should record:

- output path
- file size
- format
- sample rate
- duration
- cleanup policy

Production artifact policy is deferred until Phase 3.

## Runtime Isolation

Do not install `mlx-audio` or Qwen3-TTS dependencies into:

```text
~/Library/Application Support/MLX-Shared-Runtime/mlx-python-runtime/.venv-mlxvlm-0347900
```

Recommended Phase 1/2 isolated runtime root:

```text
~/Documents/OpenClaw Assets/Apps/MLX-TurboQuant-Service-dev/runtime/qwen3-tts-smoke/
```

Recommended model storage root:

```text
~/Documents/OpenClaw Assets/Downloads - LLM Models & OC Source Code/LLM Models/
```

The target model is not currently present locally as of the Phase 1 orientation check.

## Memory And Governor Contract

Do not use text-model KV assumptions for speech generation.

Speech backends should estimate and later report:

- model weight footprint
- MLX active memory
- MLX peak memory
- MLX cache memory
- codec/talker state if visible
- decoded audio buffers
- stream buffer size when streaming is enabled
- generated duration

Governor admission for Qwen3-TTS should start conservative until measured. Phase 1 smoke should capture observed peak memory before Phase 2 chooses a lane estimate.

## Streaming And Cancel Lifecycle

Streaming is not required for first useful integration.

Before enabling streaming, design:

- binary chunk envelope
- output format per chunk
- client disconnect behavior
- timeout behavior
- `cancel(job_id)` behavior
- partial artifact cleanup
- backpressure behavior
- metrics for first audio and total audio

If these are not designed, Phase 2 should leave streaming disabled.

## Relationship To Existing Voice Lane

The existing `4019` E2B voice lane must not be changed in Phase 1 or Phase 2.

Qwen3-TTS should initially be treated as:

- a dev-only speech generation experiment
- not a replacement for `4019`
- not part of production audio intake/transcription
- not active in OpenClaw provider routing

Phase 3 can decide whether Qwen3-TTS becomes a fast outbound voice path, an opt-in alternate, or stays experimental.

## Phase 1 Smoke Requirements

Smoke target:

```text
mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit
```

The smoke report must record:

- exact model path
- model disk size
- dependency/runtime path
- load time
- total render time
- first audio latency if streaming is available
- peak RSS if measurable
- peak MLX Metal memory if measurable
- sample rate
- output duration
- output path
- subjective quality notes
- any fallback model and why

## Phase 1 Checkpoint Criteria

Phase 1 is ready for review when these exist:

- this contract spec
- isolated runtime/dependency note
- Qwen3-TTS smoke report or documented blocker
- no production file changes
- no shared neutral venv changes
- clean dev branch status or intentional local commit(s)

Then stop for Sam review and Mauricio checkpoint before Phase 2.
