#!/usr/bin/env bash
set -euo pipefail
PIPER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PIPER_ROOT"
PIPER_MUJOCO_PYTHON="${PIPER_MUJOCO_PYTHON:-$PIPER_ROOT/.venv-mujoco/bin/python}"
export PYTHONPATH="$PIPER_ROOT/src"
export PYTHONUNBUFFERED=1
# EGL supports GPU offscreen capture on this Linux host. GLFW is retained for
# the native viewer; users may explicitly select another installed GL backend.
for arg in "$@"; do
  if [[ "$arg" == "--headless" ]]; then export MUJOCO_GL="${MUJOCO_GL:-egl}"; fi
done
exec "$PIPER_MUJOCO_PYTHON" "$PIPER_ROOT/scripts/run_scene.py" --backend mujoco "$@"
