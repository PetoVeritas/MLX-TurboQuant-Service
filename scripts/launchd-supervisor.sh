#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p "$ROOT_DIR/tmp"
export PYTHONPATH="src"
exec /opt/homebrew/opt/python@3.14/bin/python3.14 -m supervisor.main
