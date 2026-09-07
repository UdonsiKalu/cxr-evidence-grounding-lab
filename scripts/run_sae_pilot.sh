#!/usr/bin/env bash
# Run thin SAE pilot with the CUDA-capable faiss_gpu1 venv (aka faissgpu1).
# Usage: ./scripts/run_sae_pilot.sh score|causal|all [-- ...]
set -euo pipefail
LAB="$(cd "$(dirname "$0")/.." && pwd)"
# Canonical: staging/cxrlabs/faiss_gpu1  (handoff alias: faissgpu1)
PY="${CXR_PYTHON:-$LAB/../cxrlabs/faiss_gpu1/bin/python}"
if [[ ! -x "$PY" ]]; then
  # Fallback: lab HF runtime used for Track B phases
  PY="$LAB/.venv-phase9/bin/python"
fi
if [[ ! -x "$PY" ]]; then
  echo "No faiss_gpu1 or .venv-phase9 python found. Set CXR_PYTHON=" >&2
  exit 1
fi
echo "[run_sae_pilot] using $PY"
exec "$PY" -m n2s_lab.n2s_sae_pilot "$@"
