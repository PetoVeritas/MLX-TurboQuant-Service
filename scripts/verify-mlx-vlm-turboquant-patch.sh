#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_FILE="${PATCH_FILE:-$ROOT_DIR/runtime-patches/mlx-vlm-0.6.3-gemma4-elastic-kv.patch}"
VENV_DIR="${MLX_VLM_VENV:-${1:-}}"

if [[ -z "$VENV_DIR" ]]; then
  echo "usage: MLX_VLM_VENV=/path/to/venv E2B_TURBOQUANT_MODEL=/path/to/e2b TURBOQUANT_26B_MODEL=/path/to/26b $0" >&2
  echo "   or: $0 /path/to/venv" >&2
  exit 2
fi

PYTHON="$VENV_DIR/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "missing python executable: $PYTHON" >&2
  exit 2
fi

if [[ ! -f "$PATCH_FILE" ]]; then
  echo "missing patch file: $PATCH_FILE" >&2
  exit 2
fi

LANGUAGE_FILE="$("$PYTHON" - <<'PY'
import inspect
from pathlib import Path
import mlx_vlm.models.gemma4.language as language

print(Path(inspect.getfile(language)))
PY
)"

echo "mlx-vlm language file: $LANGUAGE_FILE"
echo "patch file: $PATCH_FILE"

if patch --dry-run --forward "$LANGUAGE_FILE" "$PATCH_FILE" >/dev/null 2>&1; then
  patch --forward "$LANGUAGE_FILE" "$PATCH_FILE"
elif patch --dry-run --reverse "$LANGUAGE_FILE" "$PATCH_FILE" >/dev/null 2>&1; then
  echo "patch already applied"
else
  echo "patch does not apply cleanly and is not already applied" >&2
  exit 1
fi

"$PYTHON" - <<'PY'
from importlib.metadata import version

print("mlx-vlm", version("mlx-vlm"))
print("mlx", version("mlx"))
PY

: "${E2B_TURBOQUANT_MODEL:?set E2B_TURBOQUANT_MODEL to the local E2B TurboQuant model path}"
: "${TURBOQUANT_26B_MODEL:?set TURBOQUANT_26B_MODEL to the local 26B TurboQuant model path}"

"$PYTHON" -m mlx_vlm.generate \
  --model "$E2B_TURBOQUANT_MODEL" \
  --prompt "Reply with exactly: text smoke ok" \
  --max-tokens 12 \
  --temperature 0.0

"$PYTHON" - <<'PY' "$E2B_TURBOQUANT_MODEL"
import json
import sys
from pathlib import Path
from mlx_vlm import load

model_path = Path(sys.argv[1])
config = json.loads((model_path / "config.json").read_text())
text = config["text_config"]
model, _processor = load(str(model_path))
lm = model.language_model.model
cache = model.make_cache()

expected = {
    "max_position_embeddings": 131072,
    "sliding_window": 512,
    "num_hidden_layers": 35,
    "num_kv_shared_layers": 20,
    "hidden_size_per_layer_input": 256,
    "attention_k_eq_v": False,
}
for key, value in expected.items():
    actual = text.get(key)
    if actual != value:
        raise SystemExit(f"{key} mismatch: expected {value!r}, got {actual!r}")

quant = config.get("quantization") or {}
if quant.get("bits") != 4 or quant.get("group_size") != 64 or quant.get("mode") != "affine":
    raise SystemExit(f"quantization mismatch: {quant!r}")
if len(lm.layers) != 35:
    raise SystemExit(f"loaded layer count mismatch: {len(lm.layers)}")
if getattr(lm, "first_kv_shared_layer_idx", None) != 15:
    raise SystemExit(f"first shared layer mismatch: {getattr(lm, 'first_kv_shared_layer_idx', None)!r}")
if len(cache) != 15:
    raise SystemExit(f"cache length mismatch: {len(cache)}")
for idx in (15, 34):
    attn = lm.layers[idx].self_attn
    for name in ("k_proj", "v_proj", "k_norm"):
        if not hasattr(attn, name):
            raise SystemExit(f"layer {idx} missing {name}")
print("E2B KV/context parity ok")
PY

"$PYTHON" -m mlx_vlm.generate \
  --model "$TURBOQUANT_26B_MODEL" \
  --prompt "Reply with exactly: 26b regression ok" \
  --max-tokens 10 \
  --temperature 0.0

echo "mlx-vlm TurboQuant patch verification complete"
