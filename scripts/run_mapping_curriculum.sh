#!/usr/bin/env bash
# Dual Translate mapping curriculum UI (read-only). Port 8264.
# Usage: ./scripts/run_mapping_curriculum.sh
set -euo pipefail
LAB="$(cd "$(dirname "$0")/.." && pwd)"
cd "$LAB/curriculum-mapping"
PY="${CXR_PYTHON:-$LAB/../cxrlabs/faiss_gpu1/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="$LAB/.venv-phase9/bin/python"
fi
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi
exec "$PY" server.py "$@"
