#!/usr/bin/env bash
set -euo pipefail
PIPER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PIPER_ROOT"
PIPER_PYTHON="${PIPER_PYTHON:-$PIPER_ROOT/.venv-isaac/bin/python}"
if ! "$PIPER_PYTHON" -c 'import importlib.util; assert importlib.util.find_spec("isaaclab")' 2>/dev/null; then
  echo "Isaac runtime is not ready. Run scripts/install_runtime.sh; see README.md." >&2
  exit 1
fi
# The desktop shell contains ROS Python 3.12 paths; do not inject them into Kit's 3.11.
export PYTHONPATH="$PIPER_ROOT/src"
export PYTHONUNBUFFERED=1
exec "$PIPER_PYTHON" "$PIPER_ROOT/scripts/run_scene.py" "$@"
