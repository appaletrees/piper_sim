#!/usr/bin/env bash
set -euo pipefail
PIPER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PIPER_ROOT"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--rebuild-assets" ) ]]; then
  echo "Usage: bash scripts/install_mujoco.sh [--rebuild-assets]" >&2
  exit 2
fi
command -v uv >/dev/null || { echo "Install uv before running this script." >&2; exit 1; }
PIPER_MUJOCO_PYTHON="$PIPER_ROOT/.venv-mujoco/bin/python"
if ! test -x "$PIPER_MUJOCO_PYTHON"; then uv venv --python 3.11 .venv-mujoco; fi
uv pip install --python "$PIPER_MUJOCO_PYTHON" -e '.[mujoco,test,hardware]' \
  -c requirements/mujoco-linux-py311.lock --build-constraint requirements/build-constraints.txt
export PYTHONPATH="$PIPER_ROOT/src"
if [[ "${1:-}" == "--rebuild-assets" ]]; then
  uv pip install --python "$PIPER_MUJOCO_PYTHON" -e '.[cad]'
  "$PIPER_MUJOCO_PYTHON" scripts/prepare_assets.py
  "$PIPER_MUJOCO_PYTHON" scripts/convert_workcell_cad.py
  "$PIPER_MUJOCO_PYTHON" scripts/prepare_cad_gripper.py
  "$PIPER_MUJOCO_PYTHON" scripts/prepare_mujoco.py
fi
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PIPER_MUJOCO_PYTHON" -m pytest -q
uv pip check --python "$PIPER_MUJOCO_PYTHON"
echo "Installed. Launch: bash scripts/run_mujoco.sh --enable_cameras"
