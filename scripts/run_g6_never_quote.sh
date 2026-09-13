#!/usr/bin/env bash
# Goal 6 never-quote repair — CPU on Dual-fill extracts.
# Usage: ./scripts/run_g6_never_quote.sh selftest|panel
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
echo "[run_g6_never_quote] using $PY"
case "${1:-panel}" in
  selftest) exec "$PY" -m n2s_lab.n2s_g6_never_quote --selftest ;;
  panel) exec "$PY" -m n2s_lab.n2s_g6_never_quote --panel ;;
  *) echo "usage: $0 selftest|panel" >&2; exit 2 ;;
esac
