#!/usr/bin/env bash
# U-A held-out paraphrase generalization (faiss_gpu1).
# Usage: ./scripts/run_upstream_ua_gen.sh panel|readout [...]
set -euo pipefail
LAB="$(cd "$(dirname "$0")/.." && pwd)"
PY="${CXR_PYTHON:-$LAB/../cxrlabs/faiss_gpu1/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="$LAB/.venv-phase9/bin/python"
fi
if [[ ! -x "$PY" ]]; then
  echo "No faiss_gpu1 or .venv-phase9 python found. Set CXR_PYTHON=" >&2
  exit 1
fi
echo "[run_upstream_ua_gen] using $PY"
exec "$PY" -m n2s_lab.n2s_upstream_ua_generalize "$@"
