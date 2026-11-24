#!/usr/bin/env bash
# One entry point:  ./run.sh [serve|test|eval|search D1|gen]   (default: serve)
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then python3 -m venv .venv; fi
if ! .venv/bin/python -c "import fastapi, sklearn, uvicorn" 2>/dev/null; then
  .venv/bin/pip install -q -r requirements.txt
fi
cmd="${1:-serve}"; shift || true
case "$cmd" in
  serve) exec .venv/bin/python -m priorart serve "$@" ;;
  test)  exec .venv/bin/python -m pytest -q "$@" ;;
  *)     exec .venv/bin/python -m priorart "$cmd" "$@" ;;
esac
