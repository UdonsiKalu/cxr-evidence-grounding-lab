#!/usr/bin/env bash
# V1 Dual Translate map viewer (CPU, read-only). Port 8263.
# Usage: ./scripts/run_map_viewer.sh
set -euo pipefail
LAB="$(cd "$(dirname "$0")/.." && pwd)"
cd "$LAB"
PY="${CXR_PYTHON:-$LAB/../cxrlabs/faiss_gpu1/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="$LAB/.venv-phase9/bin/python"
fi
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi
exec "$PY" map_viewer_server.py "$@"
