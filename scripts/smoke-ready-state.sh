#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${MLX_GEMMA_PORT:-4027}"
MODEL_ID="${MLX_GEMMA_MODEL_ID:-mlx-gemma-test-local}"
SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

require_json_field() {
  local file="$1"
  local expr="$2"
  python3 - "$file" "$expr" <<'PY'
import json, sys
path, expr = sys.argv[1], sys.argv[2]
with open(path, 'r', encoding='utf-8') as fh:
    data = json.load(fh)
value = data
for part in expr.split('.'):
    value = value[part]
if isinstance(value, bool):
    print('true' if value else 'false')
elif value is None:
    print('null')
else:
    print(value)
PY
}

assert_eq() {
  local actual="$1"
  local expected="$2"
  local message="$3"
  if [[ "$actual" != "$expected" ]]; then
    echo "FAIL: $message (expected '$expected', got '$actual')" >&2
    exit 1
  fi
}

cd "$ROOT_DIR"
# Export MLX_GEMMA_MODEL_ID so the supervisor we spawn and the client requests
# below agree on the same model id regardless of what's in config/local.json.
MLX_GEMMA_STUB_MODE=1 MLX_GEMMA_PORT="$PORT" MLX_GEMMA_MODEL_ID="$MODEL_ID" \
  PYTHONPATH=src python3 -m supervisor.main > /tmp/mlx-smoke-ready.log 2>&1 &
SERVER_PID="$!"

for _ in {1..50}; do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.2
done

curl -fsS "http://127.0.0.1:${PORT}/ready" > /tmp/mlx-ready-before.json
assert_eq "$(require_json_field /tmp/mlx-ready-before.json ok)" "false" "ready.ok should be false before first load"
assert_eq "$(require_json_field /tmp/mlx-ready-before.json accepting_requests)" "true" "service should still accept a cold-load request"
assert_eq "$(require_json_field /tmp/mlx-ready-before.json cold_load_acceptable)" "true" "cold-load path should be advertised before first request"

curl -fsS \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":32}" \
  "http://127.0.0.1:${PORT}/v1/chat/completions" > /tmp/mlx-chat-once.json

curl -fsS "http://127.0.0.1:${PORT}/ready" > /tmp/mlx-ready-after.json
assert_eq "$(require_json_field /tmp/mlx-ready-after.json ok)" "true" "ready.ok should be true after worker load"
assert_eq "$(require_json_field /tmp/mlx-ready-after.json actively_ready)" "true" "actively_ready should be true after worker load"
assert_eq "$(require_json_field /tmp/mlx-ready-after.json cold_load_acceptable)" "false" "cold-load flag should clear after worker load"

printf 'PASS: readiness semantics verified on port %s\n' "$PORT"
