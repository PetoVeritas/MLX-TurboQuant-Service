# MLX TurboQuant Service — Supervised, Local Gemma 4 on Apple Silicon

Runs Gemma 4 models locally on Apple Silicon via MLX and exposes them as OpenAI-compatible provider boundaries for OpenClaw-style agent stacks. A lightweight HTTP supervisor manages a separate worker process so the model stays up, restarts cleanly, and behaves predictably under agent workloads — single-target on purpose, not a generic multi-model surface.

The primary lane is the **Gemma 4 26B-A4B** TurboQuant model — the flagship, highest-priority service this stack is built around. The smaller E2B lanes run alongside it for audio-capable and voice workloads.

The backend is **`mlx-vlm`** (with a vendored elastic-KV patch for Gemma 4 E2B/E4B TurboQuant models). It supports **text, image, and audio** modalities with **tool calling** (OpenAI-compatible `tools` parameter). Model support depends on the artifact: 26B TurboQuant supports text+image; E2B TurboQuant supports text+image+audio.

## Current Production Lanes

| Port | Model | Quant | Context | Modalities | Priority |
|------|-------|-------|---------|------------|----------|
| 4017 | Gemma 4 26B A4B IT | TurboQuant 8-bit | 73728 | image, text | 1 |
| 4018 | Gemma 4 E2B IT | TurboQuant 4-bit | 73728 | audio, image, text | 2 |
| 4019 | Gemma 4 E2B IT | TurboQuant 4-bit | 16384 | audio, image, text | 3 |

All three lanes use the same `mlx-vlm` runtime with the elastic-KV patch. 4018 and 4019 run separate instances of the same E2B weights at different context windows.

## Why this exists

Getting a model to produce tokens is not enough for real agent use.
This project exists to make local Gemma 4 inference **with TurboQuant** operationally usable by adding:

- a stable OpenAI-style chat endpoint with streaming
- tool calling with hallucinated-tool containment
- multimodal inference (image, audio) via `mlx-vlm`
- supervised worker lifecycle management
- local health and admin endpoints
- smoke, recovery, and timeout testing
- a cleaner path for evaluating MLX as a serious OpenClaw lane

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
  - optional file-lock/state-file admission control for sibling services so smaller lanes do not casually crowd out the protected 26B lane
- **Operational visibility**
  - structured request and state-transition logs
  - timing metrics for load, prefill, generation, and total request time
  - diffusion backends keep autoregressive-only timings nullable and report diffusion canvas/work counters separately
- **Local testing tools**
  - smoke, recovery, timeout, fixture, soak, reclaim, and lane-comparison scripts
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
| POST | `/admin/worker/unload` | Unload the worker |
| POST | `/admin/worker/restart` | Restart the worker |

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

### Modality configuration

The `modalities` block in config controls which input types each lane accepts:

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

Effective modalities for a request are the intersection of **configured** (lane config), **backend-supported** (what the loaded model can do), and the request payload. When `strictCapabilityCheck` is `true`, requests for disabled modalities are rejected with `422 unsupported_modality`.

### Governor sizing

Recommended local shape (based on actual memory measurements):

- 26B lane: `rssEstimateLoadedGb: 29.0` (actual peak ~28.9 GB), `priority: 1`
- E2B lane: `rssEstimateLoadedGb: 6.0` (actual peak ~4.4 GB), `priority: 2` or `3`
- Shared ceiling: `ceilingGb: 34.0` (on 48 GB machines)
- Keep `allowLowerPriorityToPreemptHigher: false` so E2B lanes cannot preempt 26B by default

When a cold load would exceed the ceiling, the governor refuses admission with `governor_refused` unless a configured preemption path can safely unload lower-priority rows first.

Note: the 26B TurboQuant model uses ~29 GB at peak under `mlx-vlm`. Configuring the estimate at 20 GB (the `mlx-lm` baseline) causes incorrect admission decisions. Set it to at least 29 GB for accurate co-residency.

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

DiffusionGemma uses the same OpenAI-facing tool-call response shape as the TurboQuant lanes, but the worker obtains it by prompting the Gemma chat template and parsing the raw `mlx-vlm` output. Malformed diffusion tool-call output is treated as a retryable backend formatting failure instead of being returned as assistant prose.

## Streaming and Metrics

Autoregressive TurboQuant lanes stream incremental text tokens. DiffusionGemma denoises a canvas, so its streaming path emits only finalized output: one finalized content delta for normal text responses, or the final structured `tool_calls` delta for tool-use responses. It does not stream draft canvas text.

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

## Runtime Patch

The service requires a patched `mlx-vlm` to load Gemma 4 E2B/E4B TurboQuant models. The patch fixes `mlx_vlm/models/gemma4/language.py` to allocate K/V modules for all layers (matching `mlx-lm`'s behavior) instead of skipping layers marked as KV-shared.

The patch file is at `runtime-patches/mlx-vlm-0.6.3-gemma4-elastic-kv.patch`. It must be applied to the `mlx-vlm` package inside the worker virtualenv after installation. A verification script is provided at `scripts/verify-mlx-vlm-turboquant-patch.sh`.

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
- supervised worker startup, idle unload, and recovery
- readiness and stats inspection that stays responsive during active generation
- cold/warm request validation and fixture-based cleanliness checks
- shared memory-governor admission for sibling lanes
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
- OpenClaw-compatible workflow if used as a lane

## Development note

Built AI-assisted, using personal [OpenClaw](https://github.com/openclaw/openclaw) agents as coding collaborators.

---

**Copyright:** © 2026 PetoVeritas  
**License:** Apache-2.0
