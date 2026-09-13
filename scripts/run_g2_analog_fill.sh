#!/usr/bin/env bash
# Goal 2 — Dual analog extract fill (schema in system, user=evidence).
# Usage: ./scripts/run_g2_analog_fill.sh selftest|recover
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
echo "[run_g2_analog_fill] using $PY"
case "${1:-recover}" in
  selftest) exec "$PY" -m n2s_lab.n2s_g2_analog_fill --selftest ;;
  recover) exec "$PY" -m n2s_lab.n2s_g2_analog_fill --recover-hf --census ;;
  *) echo "usage: $0 selftest|recover" >&2; exit 2 ;;
esac
