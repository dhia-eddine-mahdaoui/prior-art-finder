#!/usr/bin/env bash
# One entry point:  ./run.sh [serve|test|eval|report|tune|search D1|gen|capture shots|capture demo]   (default: serve)
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then python3 -m venv .venv; fi
if ! .venv/bin/python -c "import fastapi, sklearn, uvicorn" 2>/dev/null; then
  .venv/bin/pip install -q -r requirements.txt
fi
cmd="${1:-serve}"; shift || true
case "$cmd" in
  serve)   exec .venv/bin/python -m priorart serve "$@" ;;
  test)    exec .venv/bin/python -m pytest -q "$@" ;;
  capture) .venv/bin/python -c "import playwright, imageio_ffmpeg, PIL" 2>/dev/null || .venv/bin/pip install -q playwright imageio-ffmpeg pillow
           exec .venv/bin/python scripts/capture.py "$@" ;;
  *)       exec .venv/bin/python -m priorart "$cmd" "$@" ;;
esac
