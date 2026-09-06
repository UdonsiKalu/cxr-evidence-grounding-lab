#!/usr/bin/env bash
# Circuit pilot via faiss_gpu1 (aka faissgpu1).
set -euo pipefail
LAB="$(cd "$(dirname "$0")/.." && pwd)"
PY="${CXR_PYTHON:-$LAB/../cxrlabs/faiss_gpu1/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="$LAB/.venv-phase9/bin/python"
fi
if [[ ! -x "$PY" ]]; then
  echo "No faiss_gpu1 or .venv-phase9. Set CXR_PYTHON=" >&2
  exit 1
fi
echo "[run_circuit_pilot] using $PY"
exec "$PY" -m n2s_lab.n2s_circuit_pilot "$@"
