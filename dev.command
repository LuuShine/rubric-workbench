#!/bin/zsh
set -eu
cd "$(dirname "$0")"
export GRADIO_SERVER_NAME=127.0.0.1
export GRADIO_ANALYTICS_ENABLED=false
export MPLCONFIGDIR="$PWD/.local/matplotlib"
exec .venv/bin/python app.py
