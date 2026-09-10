#!/usr/bin/env bash
# Locator v1 GUI (CPU). Usage: ./scripts/run_locator.sh
set -euo pipefail
LAB="$(cd "$(dirname "$0")/.." && pwd)"
cd "$LAB"
exec python3 locator_server.py "$@"
