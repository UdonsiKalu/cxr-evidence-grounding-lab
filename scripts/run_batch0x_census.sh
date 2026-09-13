#!/usr/bin/env bash
# Batch 0x expand — Dual analog + Schema census, no intervention.
# Usage: ./scripts/run_batch0x_census.sh selftest|manifest|recover
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
echo "[run_batch0x_census] using $PY"
case "${1:-recover}" in
  selftest) exec "$PY" -m n2s_lab.n2s_batch0x_census --selftest ;;
  manifest) exec "$PY" -m n2s_lab.n2s_batch0x_census --manifest ;;
  recover) exec "$PY" -m n2s_lab.n2s_batch0x_census --recover-hf --census ;;
  *) echo "usage: $0 selftest|manifest|recover" >&2; exit 2 ;;
esac
