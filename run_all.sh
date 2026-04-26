#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python3"
fi
"$PYTHON_BIN" run_all.py
