#!/usr/bin/env bash
# Batch 0 phenotype census — Dual analog 7B, no d-patch.
# Usage: ./scripts/run_batch0_phenotype.sh selftest|recover
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
echo "[run_batch0_phenotype] using $PY"
case "${1:-recover}" in
  selftest) exec "$PY" -m n2s_lab.n2s_batch0_phenotype --selftest ;;
  recover) exec "$PY" -m n2s_lab.n2s_batch0_phenotype --recover-hf --census ;;
  *) echo "usage: $0 selftest|recover" >&2; exit 2 ;;
esac
