# MLX + SI Drone TurboQuant and Diffusion Server — Process-Managed Local Gemma 4 with Stateful/Sessionized Inference Drone Service on Apple Silicon

Runs Gemma 4 models locally on Apple Silicon through a process-managed server built for TurboQuant on MLX, exposing them as OpenAI-compatible provider boundaries and short-lived SI Drone sessions (which preserve model-side multimodal traces in the inference cache between turns) for OpenClaw-style agent stacks. A lightweight HTTP supervisor manages a separate worker process so the model stays up, restarts cleanly, and behaves predictably under agent workloads — single-target on purpose, not a generic multi-model surface.

The server has been exercised across multiple local Gemma 4 model profiles:

- **Gemma 4 26B-A4B TurboQuant 8-bit** — `majentik/gemma-4-26B-A4B-it-TurboQuant-MLX-8bit`
- **Gemma 4 E2B TurboQuant 4-bit** — `majentik/gemma-4-E2B-it-TurboQuant-MLX-4bit`
- **Gemma 4 E4B TurboQuant 8-bit** — `majentik/gemma-4-E4B-TurboQuant-MLX-8bit`
- **DiffusionGemma 26B-A4B 4-bit** — `mlx-community/diffusiongemma-26B-A4B-it-4bit`

All profiles use the **`mlx-vlm`** runtime family. The autoregressive TurboQuant profiles use the Gemma4/TurboQuant backend path (with a vendored elastic-KV patch for Gemma 4 E2B/E4B TurboQuant models), while **DiffusionGemma** uses the separate `mlx-vlm` diffusion backend path, not the elastic-KV path. DiffusionGemma depends on a pinned upstream `mlx-vlm` build with native chunked prefill (`prefill_step_size: 2048`), which replaced the old dense-mask prefill path that made large prompts explode in memory; its local profile also uses the `entropy-bound` sampler to avoid the repetition loops seen with `confidence-threshold`. Both backend families share the same supervisor/worker lifecycle, OpenAI-compatible API, and tool-call response shape, while modality support still depends on the model artifact and local config: 26B TurboQuant has been run for text+image, E4B TurboQuant for text+image+audio, and DiffusionGemma is text+image only in the current local profile.

## Why this exists

Getting a model to produce tokens is not enough for real agent use.
This project exists to make local Gemma 4 inference **with TurboQuant** operationally usable by adding:

- a stable OpenAI-style chat endpoint with streaming
- tool calling with hallucinated-tool containment
- multimodal inference (image, audio) via `mlx-vlm`
- process-managed worker lifecycle
- local health and admin endpoints
- smoke, recovery, and timeout testing
- a cleaner path for evaluating MLX as a serious OpenClaw provider

## Features

- **OpenAI-style API**
  - `POST /v1/chat/completions` (streaming and non-streaming)
  - tool-call support: pass an OpenAI `tools` array; model responds with `finish_reason=tool_calls` and structured `tool_calls` in the response
  - hallucinated tool containment: calls to undeclared tools are filtered and surfaced as errors
  - configurable sampling (temperature, top-p)
  - strips model-internal reasoning markers so only the final answer reaches clients
- **Multimodal input**
  - image: PNG/JPEG/WebP via data URL
  - audio: WAV/MP3/M4A via data URL
  - modality support is declared per-model via the `modalities` config; `strictCapabilityCheck` rejects requests for disabled modalities with `422 unsupported_modality`
- **Supervisor + worker design**
  - keeps inference isolated from the control plane over a JSON-framed subprocess pipe
- **Bounded request queue**
  - allows one active worker request plus a small configurable FIFO queue instead of dropping the first overlap as `worker_busy`
- **Lifecycle controls**
  - lazy load, idle unload, explicit unload, restart, readiness, and health checks
- **Shared memory governor**
  - optional file-lock/state-file admission control for sibling services so smaller profiles do not casually crowd out a protected large model
- **Operational visibility**
  - structured request and state-transition logs
  - timing metrics for load, prefill, generation, and total request time
  - diffusion backends keep autoregressive-only timings nullable and report diffusion canvas/work counters separately
- **Local testing tools**
  - smoke, recovery, timeout, fixture, soak, reclaim, and profile-comparison scripts
- **Local-first security posture**
  - designed for loopback/private host use with enforceable local-only admin endpoints

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Basic health check |
| GET | `/ready` | Readiness and worker-state view |
| GET | `/v1/models` | Exposed model list (includes modality metadata) |
| GET | `/admin/stats` | Worker metrics and config snapshot |
| POST | `/v1/chat/completions` | OpenAI-style chat completions (SSE streaming supported) |
| POST | `/v1/si-drones` | Create a short-lived stateful SI Drone inference session |
| POST | `/v1/si-drones/{id}/turns` | Submit a text/audio turn to an SI Drone session |
| DELETE | `/v1/si-drones/{id}` | Delete an SI Drone session and release its worker cache |
| POST | `/admin/worker/unload` | Unload the worker |
| POST | `/admin/worker/restart` | Restart the worker |

### Stateful/Sessionized Inference (SI) Drone Sessions

SI Drones provide worker-pinned vRAM inference sessions for this process-managed local server. In typical cloud-hosted inference servers, each client request carries its full context history for each turn and gets scheduled across shared accelerator memory; the model does not keep a private, turn-by-turn recollection for a particular client. An SI Drone makes the opposite local tradeoff: it reserves one worker process for an explicit session that keeps its model-side state hot in GPU-resident memory and its multimodal traces warm in the inference/KV cache.

That pinned cache lets native audio or images survive across follow-up turns without rebuilding the entire multimodal prompt. This is not general chat memory; it is temporary model-side continuity. MVP tests started with 30-second audio ingestion; the current supervisor policy defaults to and caps local configuration at `audio_seconds_per_turn: 45`.

### DiffusionGemma Profile

DiffusionGemma is the server's sibling discrete-diffusion Gemma 4 profile. It still runs through the `mlx-vlm` runtime family, but it does not generate tokens through the autoregressive TurboQuant/elastic-KV path. Instead, it denoises a text canvas, which changes the memory and streaming profile: long-prompt prefill must be chunked, sampler choice matters, and streaming emits finalized output rather than incremental draft tokens.

## Project Layout

```text
mlx-turbo-gemma-service/
├── src/
│   ├── supervisor/      # HTTP control plane
│   ├── worker/          # inference worker process
│   └── shared/          # shared config, models, constants
├── config/              # default + example local override
├── scripts/             # start/stop/smoke/fixture/soak helpers
├── benchmarks/          # prompt fixtures and shared prompt text
├── runtime-patches/     # vendored patches for mlx-vlm
├── runtime/             # dedicated MLX Python virtualenv (gitignored)
├── logs/                # runtime logs (gitignored)
└── tmp/                 # scratch runtime files (gitignored)
```

## Running

Start the service:

```bash
./scripts/start
```

Check status:

```bash
./scripts/state
```

Stop or restart:

```bash
./scripts/stop
./scripts/restart
```

## Configuration

Configuration is split between:

- `config/default.json` for baseline defaults
- `config/local.example.json` for example overrides
- `config/local.json` for machine-specific model/runtime settings (gitignored)

Typical local settings include model path, model id, Python runtime path, startup/request/probe timeouts, lazy-load behavior, idle-unload behavior, governor behavior, modalities, and sampling (temperature, top-p).

The SI Drone `sessions.onOverflow` policy is reserved for future behavior; the current implementation supports reject-only overflow handling.

### Modality configuration

The `modalities` block in config controls which input types each configured profile accepts:

```json
{
  "modalities": {
    "text": { "enabled": true },
    "image": { "enabled": true, "allowedMimeTypes": ["image/png", "image/jpeg", "image/webp"], "transport": ["data_url"] },
    "audio": { "enabled": true, "allowedMimeTypes": ["audio/wav", "audio/x-wav"], "transport": ["data_url"] },
    "video": { "enabled": false },
    "document": { "enabled": false },
    "strictCapabilityCheck": true
  }
}
```

Effective modalities for a request are the intersection of **configured** (local config), **backend-supported** (what the loaded model can do), and the request payload. When `strictCapabilityCheck` is `true`, requests for disabled modalities are rejected with `422 unsupported_modality`.

### Governor sizing

Recommended local shape (based on actual memory measurements):

- 26B profile: `rssEstimateLoadedGb: 29.0` (actual peak ~28.9 GB), `priority: 1`
- E2B profile: `rssEstimateLoadedGb: 6.0` (actual peak ~4.4 GB), `priority: 2` or `3`
- Portable default ceiling: `ceilingGb: 34.0` (conservative; safe on a 48 GB box)
- Machines with more headroom can raise the ceiling and per-profile estimates in a local profile; see the config examples for worked model-profile shapes. Tune these to your own hardware - they are machine-specific, not server defaults.
- Keep `allowLowerPriorityToPreemptHigher: false` so smaller profiles cannot preempt a protected large profile by default

When a cold load would exceed the ceiling, the governor refuses admission with `governor_refused` unless a configured preemption path can safely unload lower-priority rows first.

Note: the 26B TurboQuant model uses ~29 GB at peak under `mlx-vlm`. Configuring the estimate at 20 GB (the `mlx-lm` baseline) causes incorrect admission decisions. Set it to at least 29 GB for accurate co-residency.

### SI Drone runtime notes

SI Drone sessions are explicit, bounded worker-cache sessions. Each session owns its own `prompt_cache` keyed by `session_id`, accepts only the modalities enabled for the active profile, and should be deleted when the caller is done. Expiry/deletion tears down the session cache; full RAM release still depends on normal worker idle unload or explicit unload.

Treat SI Drone cache state as temporary model-side continuity, not durable memory: callers should not rely on it after expiry, worker unload, model/runtime change, or session deletion. The default SI audio policy is `audio_seconds_per_turn: 45`; local config may lower that window, but values above 45 are clamped by the supervisor. Any production deployment should validate text and multimodal carryover against its own model artifact, modality policy, cleanup behavior, and runtime compatibility.

Validation snapshot from local MVP SI Drone experiments:

| Check | Turn pattern | Result |
|---|---|---|
| Text SI carryover | Turn 1 seeded text, later turn asked for recall without replaying turn 1. | Passed: recalled `cobalt lantern 42`. |
| Different phrase audio carryover | Turn 1 seeded WAV audio, later turn asked for recall from session cache. | Passed: recalled `Violet compass 83` with nonzero audio tokens. |
| Multi-detail audio carryover | Turn 1 seeded audio with several terms; later turn asked for one detail. | Passed: recalled `teal`; distractor/control terms included `orange`, `carrot`, `banana`. |
| Full-duration audio marker check | 35-second audio ingestion probe plus follow-up marker recall. | Passed: the 35-second marker test validated ingestion and later recall within the configured SI audio window. |
| Audio vs. text comparison | Parallel audio-seeded and text-seeded sessions used the same recall target. | Passed: audio path reported nonzero audio tokens, text path reported `0`, and both recalled `Marble Window 17`. |
| Expiry and cleanup | Test sessions were deleted after recall checks. | Passed: worker cache state was released and sibling service health checks remained OK. |

Create a session:

```bash
curl -sS -X POST http://127.0.0.1:<port>/v1/si-drones
```

Submit a turn:

```bash
curl -sS -X POST http://127.0.0.1:<port>/v1/si-drones/sidr_example/turns \
  -H 'Content-Type: application/json' \
  -d '{"parts":[{"type":"text","text":"Remember this code: amber seven."}],"max_tokens":32}'
```

Supported v1 turn parts:

- `{"type":"text","text":"..."}` for text
- `{"type":"audio","audio":{"format":"wav","data":"<base64-wav>"}}` for WAV audio

SI Drone usage metrics report `prompt_tokens` for the new turn input, including model-side audio placeholder tokens. `metrics.audio_token_count` breaks out the audio placeholders when present.

Delete a session:

```bash
curl -sS -X DELETE http://127.0.0.1:<port>/v1/si-drones/sidr_example
```

### DiffusionGemma runtime notes

DiffusionGemma depends on native chunked prefill from pinned upstream `mlx-vlm` commit `a0578772e92409be880543c1d26d04fd00d840dc`. The original local `mlx-vlm==0.6.3` path built dense prompt-length-squared attention masks during prefill, so large real agent prompts created massive transient memory spikes even though the model weights themselves were much smaller. The pinned upstream path exposes `prefill_step_size`, and the local profile uses `prefill_step_size: 2048` so prefill runs in query chunks instead of materializing the whole attention grid at once.

The other critical DiffusionGemma fix is sampler choice. `confidence-threshold` looked plausible but caused token lock-in and repetition loops on large agent prompts. The safe server-native default is `diffusion_sampler: "entropy-bound"` alongside `prefill_step_size: 2048`. A plain reinstall from PyPI `mlx-vlm==0.6.3` can silently remove the native chunked-prefill support even though the package metadata still reports `0.6.3`.

## Tool Calling

The service supports OpenAI-compatible tool calling. Pass a `tools` array in the request:

```json
{
  "model": "gemma-local-mlx-turboquant-26b-a4b-8bit",
  "messages": [{"role": "user", "content": "What's the weather in DC?"}],
  "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
  "max_tokens": 32
}
```

The model responds with `finish_reason: "tool_calls"` and structured `tool_calls` in the message. Multi-turn tool-call conversations (where the assistant has prior `tool_calls` in history) are supported — the service decodes tool-call arguments before passing to the Gemma chat template.

Hallucinated tool calls (calls to functions not declared in the `tools` array) are filtered out and surfaced as an error message in the response content.

DiffusionGemma uses the same OpenAI-facing tool-call response shape as the TurboQuant profiles, but the worker obtains it by prompting the Gemma chat template and parsing the raw `mlx-vlm` output. Malformed diffusion tool-call output is treated as a retryable backend formatting failure instead of being returned as assistant prose.

## Streaming and Metrics

Autoregressive TurboQuant profiles stream incremental text tokens. DiffusionGemma denoises a canvas, so its streaming path emits only finalized output: one finalized content delta for normal text responses, or the final structured `tool_calls` delta for tool-use responses. It does not stream draft canvas text.

For diffusion responses, `prefill_ms` and `generation_ms` remain `null` because those are autoregressive timing buckets. Diffusion-specific fields are reported in `metrics` when available from `mlx-vlm`:

- `first_visible_output_ms`
- `diffusion_canvas_tokens`
- `diffusion_denoising_steps`
- `diffusion_work_tokens`
- `diffusion_canvas_tps`
- `diffusion_work_tps`
- `diffusion_tool_retry_count`

## Request Queue

The supervisor intentionally keeps inference single-worker and local-first. It can absorb a bounded amount of overlap with `worker.queue.maxDepth`:

- `0`: no queue; overlapping requests are rejected as `worker_busy`.
- `1`: one active request plus one queued request.

Queued requests wait for the active request to finish, then run through the same worker path. If the queue is already full, `POST /v1/chat/completions` returns `409 queue_full`.

## Runtime Patches and Pins

The service requires a patched `mlx-vlm` to load Gemma 4 E2B/E4B TurboQuant models. The patch fixes `mlx_vlm/models/gemma4/language.py` to allocate K/V modules for all layers (matching `mlx-lm`'s behavior) instead of skipping layers marked as KV-shared.

The patch file is at `runtime-patches/mlx-vlm-0.6.3-gemma4-elastic-kv.patch`. It must be applied to the `mlx-vlm` package inside the worker virtualenv after installation. A verification script is provided at `scripts/verify-mlx-vlm-turboquant-patch.sh`.

DiffusionGemma profiles additionally require `mlx-vlm` from pinned upstream commit `a0578772e92409be880543c1d26d04fd00d840dc` so `generate/diffusion.py`, `generate/common.py`, and `models/diffusion_gemma/language.py` expose native `prefill_step_size` chunking. A plain reinstall from PyPI `mlx-vlm==0.6.3` can silently remove that support even though the package metadata still reports `0.6.3`.

**Every `mlx-vlm` upgrade must retest against E2B/E4B/26B load matrix.** The patch is version-specific and carries a maintenance burden.

## KV-cache recommendation

For TurboQuant / KV-cache experiments, the current recommendation is asymmetric compression:

- keep **K** high precision by default
- compress **V** first if memory pressure requires it
- avoid symmetric low-bit K/V compression as the default
- validate any KV change with tiny factual, long-context retrieval, tool-call, and reclaim tests before using it for agent traffic

## Helper Scripts

- `scripts/start` / `scripts/stop` / `scripts/restart`
- `scripts/state`
- `scripts/smoke-test`
- `scripts/smoke-ready-state.sh`
- `scripts/smoke-e2b-runtime-parity.py`
- `scripts/recovery-test`
- `scripts/timeout-failure-test`
- `scripts/list-fixtures`
- `scripts/run-fixture`
- `scripts/soak-profile`
- `scripts/memory-profile`
- `scripts/reclaim-profile`
- `scripts/compare-lanes`
- `scripts/verify-mlx-vlm-turboquant-patch.sh`

## Current Status

This project has moved beyond scaffold-only bring-up and into real local MLX inference testing.
It currently supports:

- real Gemma completions through `mlx-vlm` with TurboQuant models
- multimodal inference (image, audio) via data URL
- tool calling with hallucinated-tool containment
- streaming responses with channel-markup containment
- process-managed worker startup, idle unload, and recovery
- readiness and stats inspection that stays responsive during active generation
- cold/warm request validation and fixture-based cleanliness checks
- shared memory-governor admission for sibling model profiles
- early hardening for OpenClaw compatibility

## Security

- local-first by design
- intended for loopback/private use
- admin endpoints enforced local-only when `server.adminLocalOnly` is enabled
- no built-in public auth layer

## Requirements

- Apple Silicon Mac (48 GB unified memory recommended for 26B + E2B co-residency)
- Python 3.11+ (3.13/3.14 tested) with an `mlx-vlm`-capable virtualenv (see `runtime-patches/`)
- local Gemma 4 TurboQuant model files from the `majentik` org on HuggingFace
- OpenClaw-compatible workflow if used as a local provider

## Development note

Built AI-assisted, using personal [OpenClaw](https://github.com/openclaw/openclaw) agents as coding contributors (powered by GPT-5.5 Codex, Claude Opus 4.6, and Claude Opus 4.8).

---

**Copyright:** © 2026 PetoVeritas  
**License:** Apache-2.0
