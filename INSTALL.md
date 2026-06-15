# INSTALL

This project is a local MLX-backed inference service for Apple Silicon Macs.
It is not a one-click app. A new machine needs four things:

1. the repository code
2. a local Python runtime for the worker (with patched `mlx-vlm`)
3. local model files
4. a machine-specific `config/local.json`

## Requirements

- **Apple Silicon Mac.** Intel Macs are not supported.
- **Unified memory depends on the model.** 48 GB is recommended for running 26B and E2B lanes concurrently. 16 GB works for E2B alone. The 26B model peaks at ~29 GB under `mlx-vlm`.
- **Python 3.13+** (tested on 3.13 and 3.14). Minimum supported is 3.11. If you don't already have it, install via Homebrew: `brew install python@3.14`.
- **~15 GB of free disk space** per TurboQuant model variant (more for multiple lanes).
- **HuggingFace CLI** for model downloads: `pip install "huggingface_hub[cli]"`.

## What is in GitHub vs what is local

Tracked in the repo:
- service source code under `src/`
- helper scripts under `scripts/`
- default config and example local config
- benchmark fixtures
- vendored runtime patch under `runtime-patches/`
- README / docs

Not tracked in the repo:
- `runtime/`
- `logs/`
- `tmp/`
- `config/local.json`
- downloaded model weights

## 1. Clone the repo

```bash
git clone <repo-url>
cd mlx-turbo-gemma-service
```

## 2. Create the worker Python environment

The supervisor is standard-library Python and is started by `./scripts/start` using `python3`.
The worker is separate and uses the Python executable configured in `config/local.json`.

Create a virtual environment for the worker:

```bash
# Adjust Python path for your install. Homebrew on Apple Silicon:
/opt/homebrew/opt/python@3.14/bin/python3.14 -m venv runtime/mlx-python-runtime/.venv
source runtime/mlx-python-runtime/.venv/bin/activate
python --version   # should print 3.14.x (or 3.13.x)
```

Install the worker runtime dependency:

```bash
pip install --upgrade pip
pip install mlx-vlm==0.6.3
```

The service is tested with `mlx-vlm==0.6.3` and `mlx==0.31.2`. Later versions may work but require retesting the elastic-KV patch.

### Apply the elastic-KV patch

Gemma 4 E2B and E4B TurboQuant models fail to load under unpatched `mlx-vlm` because elastic (KV-shared) layers ship per-layer K/V weights that the sanitizer drops. The patch allocates those modules for all layers, matching `mlx-lm`'s behavior.

```bash
cd runtime/mlx-python-runtime/.venv/lib/python3.14/site-packages/
patch -p1 < ../../../../../runtime-patches/mlx-vlm-0.6.3-gemma4-elastic-kv.patch
```

Verify the patch applied:

```bash
../../../../../scripts/verify-mlx-vlm-turboquant-patch.sh
```

**Every `mlx-vlm` upgrade must retest against E2B/E4B/26B load matrix.** The patch is version-specific.

## 3. Download the model

The production lanes use TurboQuant models from the `majentik` org on HuggingFace:

```bash
# 26B (port 4017)
huggingface-cli download \
  majentik/gemma-4-26B-A4B-it-TurboQuant-MLX-8bit \
  --local-dir ~/models/majentik-gemma-4-26b-a4b-it-turboquant-mlx-8bit

# E2B (ports 4018, 4019)
huggingface-cli download \
  majentik/gemma-4-E2B-it-TurboQuant-MLX-4bit \
  --local-dir ~/models/majentik-gemma-4-E2B-it-TurboQuant-MLX-4bit
```

If the download fails with a 401 or 403, the repo is gated — run `huggingface-cli login`, accept the license on the HuggingFace web UI, then retry.

## 4. Create `config/local.json`

Start from the example file:

```bash
cp config/local.example.json config/local.json
```

Then edit `config/local.json`. At minimum, set:

```json
{
  "model": {
    "id": "gemma-local-mlx-turboquant-26b-a4b-8bit",
    "path": "/absolute/path/to/majentik-gemma-4-26b-a4b-it-turboquant-mlx-8bit",
    "contextWindowTokens": 73728,
    "maxOutputTokens": 8192
  },
  "worker": {
    "backend": "mlx_vlm_turboquant",
    "pythonExecutable": "/absolute/path/to/runtime/mlx-python-runtime/.venv/bin/python"
  },
  "modalities": {
    "text": { "enabled": true },
    "image": { "enabled": true, "allowedMimeTypes": ["image/png", "image/jpeg", "image/webp"], "transport": ["data_url"] },
    "audio": { "enabled": true, "allowedMimeTypes": ["audio/wav", "audio/x-wav"], "transport": ["data_url"] },
    "strictCapabilityCheck": true
  },
  "governor": {
    "enabled": true,
    "instanceId": "mlx-26b",
    "priority": 1,
    "rssEstimateLoadedGb": 29.0,
    "ceilingGb": 34.0
  }
}
```

Important fields:
- `model.id`: model name exposed to clients
- `model.path`: absolute path to the downloaded model directory
- `model.contextWindowTokens`: context window (73728 for 26B/E2B at full, 16384 for voice E2B)
- `model.maxOutputTokens`: max output tokens (default 8192)
- `worker.pythonExecutable`: absolute path to the worker venv Python
- `worker.backend`: must be `mlx_vlm_turboquant`
- `modalities`: enable image/audio as needed per lane; `strictCapabilityCheck` rejects undeclared modalities with 422

Useful optional fields:
- `server.port`: default is `4017`
- `worker.lazyLoad`: default `true`
- `worker.idleUnload.enabled`: default `true`
- `worker.idleUnload.idleMs`: default `300000`
- `governor`: shared memory admission control (see README)

## 5. Start the service

```bash
./scripts/start
```

This starts the supervisor in the background and writes runtime files under `tmp/`.

## 6. Verify that it is running

Check readiness and stats:

```bash
./scripts/state
```

Basic health:

```bash
curl http://127.0.0.1:4017/health
```

Model list (includes modality metadata):

```bash
curl http://127.0.0.1:4017/v1/models
```

Run the smoke test:

```bash
./scripts/smoke-test
```

**First-request latency.** With `worker.lazyLoad` at its default (`true`), the very first request triggers the model load, which can take 15–30 seconds. Subsequent requests are fast.

## 7. Stop or restart

```bash
./scripts/stop
./scripts/restart
```

## Environment-variable overrides

The service supports these environment variables:

- `MLX_GEMMA_HOST`
- `MLX_GEMMA_PORT`
- `MLX_GEMMA_MODEL_PATH`
- `MLX_GEMMA_MODEL_ID`
- `MLX_GEMMA_LOG_LEVEL`
- `MLX_GEMMA_STUB_MODE`
- `MLX_GEMMA_WORKER_PYTHON`

## Troubleshooting

### `mlx_model_path_not_configured`
Set `model.path` in `config/local.json`.

### `mlx_model_path_missing:...`
The configured model path does not exist on disk. Check the absolute path and that the model download completed fully.

### `mlx_runtime_import_failed:...`
The worker Python environment does not have a working `mlx-vlm` install. Confirm `worker.pythonExecutable` points at the right venv and re-run `pip install mlx-vlm==0.6.3` inside it. Make sure the elastic-KV patch is applied (see step 2).

### Service starts but `/ready` is not OK
Check in this order:
- `config/local.json`
- model path
- worker Python path
- `tmp/supervisor.log`

### Port conflict (`address already in use`)
Something else is bound to the port. Either stop the other process or change `server.port` in `config/local.json`.

### Worker keeps restarting
Tail `tmp/supervisor.log` and grep for `worker_terminated` or tracebacks. Common causes: OOM during model load, bad model path, or an `mlx-vlm` version mismatch.

### OOM during model load
The 26B TurboQuant 8-bit model needs ~29 GB of unified memory under `mlx-vlm` at peak. On a 32 GB Mac it will be tight. On a 48 GB Mac, 26B and one E2B lane can co-reside. Either run on a larger machine or use the E2B model alone.

### `governor_refused:need=Xgb ceiling=Ygb`
The shared memory governor refused admission because loading the model would exceed the configured ceiling. Options:
- Increase `ceilingGb` in `config/local.json` (ensure it stays under physical memory).
- Lower `rssEstimateLoadedGb` (only if the estimate is too conservative; 26B should be ~29 GB, E2B ~6 GB).
- Unload a lower-priority lane first via `/admin/worker/unload`.

### Tool calls return `unsupported_modality`
The `tools` parameter requires the model to support tool calling. All current TurboQuant models support it. If you see this error, check that the backend is `mlx_vlm_turboquant` (not an older `mlx_lm` backend).

### `huggingface-cli download` fails with 401 / 403
The model repo is gated. Run `huggingface-cli login`, accept the license on the HuggingFace web UI, and retry.

## Reproducibility note

The repository is enough to share the app itself, but a fresh machine still needs:
- a local worker virtualenv with `mlx-vlm==0.6.3` and the elastic-KV patch applied
- local model files
- local config

If you want fully reproducible setup, the next improvement should be a pinned `requirements.txt` for the worker environment plus an automated patch-apply step in setup.