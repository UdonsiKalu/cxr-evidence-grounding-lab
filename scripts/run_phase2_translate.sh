#!/usr/bin/env bash
# Phase 2 Translate contrast (faiss_gpu1). No d-patch.
# Usage: ./scripts/run_phase2_translate.sh selftest|recover|recover-analog|panel|panel-clean
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
echo "[run_phase2_translate] using $PY"
case "${1:-panel}" in
  selftest) exec "$PY" -m n2s_lab.n2s_phase2_translate --selftest ;;
  recover) exec "$PY" -m n2s_lab.n2s_phase2_translate --recover-hf --panel ;;
  recover-analog) exec "$PY" -m n2s_lab.n2s_phase2_translate --recover-hf-analog --reconcile ;;
  panel) exec "$PY" -m n2s_lab.n2s_phase2_translate --panel ;;
  panel-clean) exec "$PY" -m n2s_lab.n2s_phase2_translate --panel-clean ;;
  *) echo "usage: $0 selftest|recover|recover-analog|panel|panel-clean" >&2; exit 2 ;;
esac
