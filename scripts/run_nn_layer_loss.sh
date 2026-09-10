#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 nn_layer_loss_server.py "$@"
