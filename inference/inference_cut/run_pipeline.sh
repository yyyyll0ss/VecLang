#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${VECLANG_PYTHON:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python is not available: ${PYTHON_BIN}" >&2
  echo "Activate the inference environment or set VECLANG_PYTHON." >&2
  exit 2
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_pipeline.py" "$@"
