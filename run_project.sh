#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

echo "[1/4] Checking project directory..."
cd "$PROJECT_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[2/4] Creating virtual environment..."
  python3 -m venv "$VENV_DIR"
else
  echo "[2/4] Virtual environment already exists."
fi

echo "[3/4] Installing/updating dependencies..."
"$VENV_DIR/bin/pip" install -r requirements.txt

echo "[4/4] Starting Streamlit app..."
exec "$VENV_DIR/bin/streamlit" run app.py
