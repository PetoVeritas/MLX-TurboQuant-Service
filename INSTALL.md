# INSTALL

This project is a local MLX-backed inference service for Apple Silicon Macs.
It is not a one-click app yet. A new machine needs four things:

1. the repository code
2. a local Python runtime for the worker
3. local model files
4. a machine-specific `config/local.json`

## Requirements

- Apple Silicon Mac
- Python 3.11 or newer
- local model files for a supported MLX Gemma model
- a Python environment that can import `mlx_lm`

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

Create a virtual environment for the worker:

```bash
python3 -m venv runtime/mlx-python-runtime/.venv
source runtime/mlx-python-runtime/.venv/bin/activate
```

Install the worker dependency that the code expects:

```bash
pip install mlx-lm
```

Notes:
- The worker imports `mlx_lm` directly.
- If your local MLX setup requires extra packages or a specific `mlx-lm` version, install those in this same virtual environment.
- The repo does not currently pin worker dependencies in a lockfile, so if you want reproducible installs across machines, add that separately.

## 3. Download the model

Download a local MLX Gemma model to a directory on disk.
The current README references:

- `mlx-community/gemma-4-26b-a4b-it-4bit`

You will point `config/local.json` at the local path for that model.

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
The configured model path does not exist on disk.

### `mlx_runtime_import_failed:...`
The worker Python environment does not have a working `mlx_lm` install.

### Service starts but `/ready` is not OK
Check:
- `config/local.json`
- model path
- worker Python path
- `tmp/supervisor.log`

### Port conflict
Change `server.port` in `config/local.json`.

## Reproducibility note

The repository is enough to share the app itself, but a fresh machine still needs:
- a local worker virtualenv
- local model files
- local config

If you want fully reproducible setup for other people, the next improvement should be a pinned dependency file for the worker environment plus a tighter first-run setup section in `README.md`.
