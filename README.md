# MLX TurboQuant Service — Supervised, Local Gemma 4 26B on Apple Silicon

Runs Gemma 4 26B-A4B locally on Apple Silicon via MLX and exposes it as an OpenAI-compatible provider boundary for OpenClaw-style agent stacks. A lightweight HTTP supervisor manages a separate worker process so the model stays up, restarts cleanly, and behaves predictably under agent workloads — single-target on purpose, not a generic multi-model surface.

Current 26B setup note: the main service currently runs the 8-bit MLX weights at [`majentik/gemma-4-26B-A4B-it-TurboQuant-MLX-8bit`](https://huggingface.co/majentik/gemma-4-26B-A4B-it-TurboQuant-MLX-8bit) on port `4017`. In this repository, “TurboQuant” refers to the runtime/service path and KV-cache experimentation around that model family, not to a separate published 26B TQPlus weight artifact.

Current E4B setup note: the sibling test service runs [`mlx-community/gemma-4-e4b-it-8bit`](https://huggingface.co/mlx-community/gemma-4-e4b-it-8bit) on port `4018` for small-model agent usage.

## Why this exists

Getting a model to produce tokens is not enough for real agent use.
This project exists to make local Gemma 4 inference **with TurboQuant** operationally usable by adding:

- a stable OpenAI-style chat endpoint with streaming
- supervised worker lifecycle management
- local health and admin endpoints
- smoke, recovery, and timeout testing
- a cleaner path for evaluating MLX as a serious OpenClaw lane

## Features

- **OpenAI-style API**
  - `POST /v1/chat/completions` (streaming and non-streaming)
  - tool-call passthrough with hallucinated-tool containment
  - configurable sampling (temperature, top-p)
  - strips model-internal reasoning markers so only the final answer reaches clients
- **Supervisor + worker design**
  - keeps inference isolated from the control plane over a JSON-framed subprocess pipe
- **Lifecycle controls**
  - lazy load, idle unload, explicit unload, restart, readiness, and health checks
- **Shared memory governor**
  - optional file-lock/state-file admission control for sibling services so smaller lanes do not casually crowd out the protected 26B lane
- **Operational visibility**
  - structured request and state-transition logs
  - timing metrics for load, prefill, generation, and total request time
- **Local testing tools**
  - smoke, recovery, timeout, fixture, soak, reclaim, and lane-comparison scripts
- **Local-first security posture**
  - designed for loopback/private host use with enforceable local-only admin endpoints

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Basic health check |
| GET | `/ready` | Readiness and worker-state view |
| GET | `/v1/models` | Exposed model list |
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

Typical local settings include model path, model id, Python runtime path, startup/request/probe timeouts, lazy-load behavior, idle-unload behavior, governor behavior, and sampling (temperature, top-p).

Note: `model.maxOutputTokens` defaults to **8192** (raised from the previous 1024) so longer agent turns and tool-call sequences fit without per-request overrides. Lower it in `config/local.json` if you need to cap output for memory or latency reasons.

## Shared memory governor

The optional governor reserves estimated loaded-worker RSS in a JSON state file under `governor.stateDir`, protected by an advisory file lock. It is designed for sibling single-model services, not dynamic routing inside one supervisor.

Recommended local shape:

- 26B: `governor.instanceId: "mlx-26b"`, `priority: 1`, `rssEstimateLoadedGb: 20.0`
- E4B sibling: `governor.instanceId: "mlx-e4b"`, `priority: 2`, `rssEstimateLoadedGb: 12.0`
- Shared ceiling: `ceilingGb: 32.0`
- Keep `allowLowerPriorityToPreemptHigher: false` so the E4B lane cannot preempt 26B by default

When a cold load would exceed the ceiling, the governor refuses admission with `governor_refused` unless a configured preemption path can safely unload lower-priority rows first. Set `governor.enabled: false` to return to independent service behavior.

## KV-cache recommendation

For TurboQuant / KV-cache experiments, the current recommendation is asymmetric compression:

- keep **K** high precision by default
- compress **V** first if memory pressure requires it
- avoid symmetric low-bit K/V compression as the default
- validate any KV change with tiny factual, long-context retrieval, heavy-context retrieval, tool-call, and reclaim tests before using it for agent traffic

The service does **not** currently expose stable KV tuning environment variables. Add documented knobs only when the underlying MLX cache path is wired and tested; until then, prefer known-good baseline cache behavior.

## Helper Scripts

- `scripts/start` / `scripts/stop` / `scripts/restart`
- `scripts/state`
- `scripts/smoke-test`
- `scripts/smoke-ready-state.sh`
- `scripts/recovery-test`
- `scripts/timeout-failure-test`
- `scripts/list-fixtures`
- `scripts/run-fixture`
- `scripts/soak-profile`
- `scripts/memory-profile`
- `scripts/reclaim-profile`
- `scripts/compare-lanes`

## Current Status

This project has moved beyond scaffold-only bring-up and into real local MLX inference testing.
It currently supports:

- real Gemma completions through MLX with configurable sampling
- streaming responses with channel-markup and tool-call containment
- supervised worker startup, idle unload, and recovery
- readiness and stats inspection that stays responsive during active generation
- cold/warm request validation and fixture-based cleanliness checks
- optional shared memory-governor admission for 26B/E4B sibling services
- early hardening for OpenClaw compatibility

## Security

- local-first by design
- intended for loopback/private use
- admin endpoints enforced local-only when `server.adminLocalOnly` is enabled
- no built-in public auth layer

## Requirements

- Apple Silicon Mac
- Python 3.11+ with an MLX-capable virtualenv (see `runtime/`)
- local Gemma model files (for example, the 26B service can use [`majentik/gemma-4-26B-A4B-it-TurboQuant-MLX-8bit`](https://huggingface.co/majentik/gemma-4-26B-A4B-it-TurboQuant-MLX-8bit), while a smaller sibling service can use [`mlx-community/gemma-4-e4b-it-8bit`](https://huggingface.co/mlx-community/gemma-4-e4b-it-8bit); set `model.path` in `config/default.json` or a local override)
- OpenClaw-compatible workflow if used as a lane

## Development note

Built AI-assisted, using my personal [OpenClaw](https://github.com/openclaw/openclaw) agents as my coding collaborators.

---

**Copyright:** © 2026 PetoVeritas  
**License:** Apache-2.0

