#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
command -v brew >/dev/null || { echo "Install Homebrew first: https://brew.sh"; exit 1; }
brew install ffmpeg python@3.12
PY="$(brew --prefix python@3.12)/bin/python3.12"
"$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
mkdir -p models data
echo "Install complete. Run ./start.sh"
