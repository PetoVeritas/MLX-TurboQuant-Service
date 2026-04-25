# INSTALL

This project is a local MLX-backed inference service for Apple Silicon Macs.
It is not a one-click app. A new machine needs four things:

1. the repository code
2. a local Python runtime for the worker
3. local model files
4. a machine-specific `config/local.json`

## Requirements

- **Apple Silicon Mac.** Intel Macs are not supported.
- **Unified memory depends on the model.** 16GB can work for smaller 2B/4B-class MLX models. The default single 26B service is tight on 32GB but workable with headroom management. Running both the 26B service and an E4B sibling service at the same time should be treated as a 48GB+ setup.
- **Python 3.14.** Tested on 3.14.4. Minimum supported is 3.11, but recent 3.13 / 3.14 patch versions are what's actively tested. If you don't already have 3.14, install it via Homebrew: `brew install python@3.14`.
- **~20GB of free disk space** for the default model. More if you plan to try other variants.
- **HuggingFace CLI** for the model download step: `pip install "huggingface_hub[cli]"`.

## What is in GitHub vs what is local

Tracked in the repo:
- service source code under `src/`
- helper scripts under `scripts/`
- default config and example local config
- benchmark fixtures
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

Create a virtual environment for the worker, using Python 3.14:

```bash
# Path below assumes Homebrew on Apple Silicon. Adjust if installed elsewhere.
/opt/homebrew/opt/python@3.14/bin/python3.14 -m venv runtime/mlx-python-runtime/.venv
source runtime/mlx-python-runtime/.venv/bin/activate
python --version   # should print 3.14.x
```

Install the worker's runtime dependency:

```bash
pip install --upgrade pip
pip install mlx-lm
```

The service is currently tested with `mlx-lm==0.31.2`. Later versions will likely work, but if you hit a regression after upgrading, pinning to the tested version is a good first troubleshooting step.

Notes:
- The worker imports `mlx_lm` directly.
- The repo does not currently ship a pinned dependency file. If you want reproducible installs across machines, generate one with `pip freeze > requirements.txt` after a known-good install.

## 3. Download the model

The default model is `mlx-community/gemma-4-26b-a4b-it-4bit` (~15GB on disk). Download it to any directory you control:

```bash
huggingface-cli download \
  mlx-community/gemma-4-26b-a4b-it-4bit \
  --local-dir ~/models/gemma-4-26b-a4b-it-4bit
```

If the download fails with a 401 or 403, the repo is gated — run `huggingface-cli login` first, accept the license on the HuggingFace web UI for that model, then retry.

You will point `config/local.json` at the absolute path of the downloaded directory in the next step.

## 4. Create `config/local.json`

Start from the example file:

```bash
cp config/local.example.json config/local.json
```

Then edit `config/local.json`.
At minimum, set these fields:

```json
{
  "model": {
    "id": "your-model-id",
    "path": "/absolute/path/to/your/model"
  },
  "worker": {
    "pythonExecutable": "/absolute/path/to/runtime/mlx-python-runtime/.venv/bin/python"
  }
}
```

Important fields:
- `model.id`: the model name the service will expose to clients
- `model.path`: absolute path to the downloaded local model directory
- `worker.pythonExecutable`: absolute path to the worker venv Python

Useful optional fields:
- `server.port`: default is `4017`
- `worker.lazyLoad`: default `true`
- `worker.idleUnload.enabled`: default `true`
- `worker.idleUnload.idleMs`: default `300000`
- `worker.startupTimeoutMs`
- `worker.requestTimeoutMs`
- `model.maxOutputTokens`
- `model.sampling.temperature`
- `model.sampling.topP`
- `logging.level`

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

Model list:

```bash
curl http://127.0.0.1:4017/v1/models
```

Run the smoke test:

```bash
./scripts/smoke-test
```

**First-request latency.** With `worker.lazyLoad` at its default (`true`), the very first request after startup triggers the model load, which can take 15–30 seconds on an M-series Mac depending on the model and disk speed. The smoke test may appear to hang briefly on its first call — that's expected. Subsequent requests are fast.

## 7. Stop or restart

```bash
./scripts/stop
./scripts/restart
```

## Environment-variable overrides

Instead of hardcoding some values in `config/local.json`, the service also supports these environment variables:

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
The configured model path does not exist on disk. Double-check the absolute path and that the model download completed fully.

### `mlx_runtime_import_failed:...`
The worker Python environment does not have a working `mlx_lm` install. Confirm `worker.pythonExecutable` points at the right venv and re-run `pip install mlx-lm` inside it.

### Service starts but `/ready` is not OK
Check in this order:
- `config/local.json`
- model path
- worker Python path
- `tmp/supervisor.log`

### Port conflict (`address already in use`)
Something else is bound to port 4017. Either stop the other process or change `server.port` in `config/local.json`.

### Worker keeps restarting
Tail `tmp/supervisor.log` and grep for `worker_terminated` or tracebacks. Most common causes: OOM during model load, a bad model path, or an `mlx-lm` version mismatch.

### OOM during model load
The default 26B 4-bit model needs roughly 15GB of unified memory for weights, plus overhead for KV cache and macOS. On a 16GB Mac it will OOM. Either run on a larger machine, or switch to a smaller model (a 2B or 4B MLX variant) and update `model.id` / `model.path` in `config/local.json` accordingly.

### `huggingface-cli download` fails with 401 / 403
The model repo is gated. Run `huggingface-cli login`, accept the license on the HuggingFace web UI for that model, and retry the download.

## Reproducibility note

The repository is enough to share the app itself, but a fresh machine still needs:
- a local worker virtualenv
- local model files
- local config

If you want fully reproducible setup for other people, the next improvement should be a pinned `requirements.txt` for the worker environment plus a tighter first-run setup section in `README.md`.
