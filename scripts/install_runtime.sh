#!/usr/bin/env bash
# Idempotent local installation; never modifies an existing conda or system environment.
set -euo pipefail
PIPER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PIPER_ROOT"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-8}"
command -v uv >/dev/null || { echo "Install uv before running this script." >&2; exit 1; }
test -x .venv-isaac/bin/python || uv venv --python 3.11 .venv-isaac
PIPER_PYTHON="$PIPER_ROOT/.venv-isaac/bin/python"
PIPER_LAB="$PIPER_ROOT/.runtime/IsaacLab-v2.3.0"
mkdir -p "$PIPER_ROOT/.runtime/wheels"
if ! test -f "$PIPER_LAB/source/isaaclab/setup.py"; then
  mkdir -p .runtime
  curl --fail --location --retry 3 \
    https://codeload.github.com/isaac-sim/IsaacLab/tar.gz/refs/tags/v2.3.0 \
    -o .runtime/isaaclab-v2.3.0.tar.gz
  mkdir -p "$PIPER_LAB"
  tar -xzf .runtime/isaaclab-v2.3.0.tar.gz -C "$PIPER_LAB" --strip-components=1
fi
uv pip install --python "$PIPER_PYTHON" \
  'torch==2.7.0+cu128' 'torchvision==0.22.0+cu128' \
  -c requirements/runtime-linux-py311.lock \
  --find-links "$PIPER_ROOT/.runtime/wheels" --torch-backend cu128
# Prefer PyPI for shared dependencies, but require platform-specific Isaac wheels.
# The PyPI placeholder sdists may expose Windows-only metadata on Linux.
PIPER_BINARY_NAMES="isaacsim,isaacsim-app,isaacsim-asset,isaacsim-benchmark,isaacsim-code-editor,isaacsim-core,isaacsim-cortex,isaacsim-example,isaacsim-extscache-kit,isaacsim-extscache-kit-sdk,isaacsim-extscache-physics,isaacsim-gui,isaacsim-kernel,isaacsim-replicator,isaacsim-rl,isaacsim-robot,isaacsim-robot-motion,isaacsim-robot-setup,isaacsim-ros1,isaacsim-ros2,isaacsim-sensor,isaacsim-storage,isaacsim-template,isaacsim-test,isaacsim-utils"
uv pip install --python "$PIPER_PYTHON" \
  'isaacsim[all,extscache]==5.1.0' \
  --index https://pypi.org/simple --default-index https://pypi.nvidia.com \
  --index-strategy unsafe-best-match --only-binary "$PIPER_BINARY_NAMES" \
  -c requirements/runtime-linux-py311.lock \
  --torch-backend cu128 --find-links "$PIPER_ROOT/.runtime/wheels"
uv pip install --python "$PIPER_PYTHON" -e "$PIPER_LAB/source/isaaclab" -e "$PIPER_ROOT[test,hardware]" \
  -c requirements/runtime-linux-py311.lock --build-constraint requirements/build-constraints.txt \
  --torch-backend cu128
PYTHONPATH="$PIPER_ROOT/src" "$PIPER_PYTHON" scripts/prepare_assets.py
if ! test -f "$PIPER_ROOT/assets/generated/workcell/camera_rig.json"; then
  test -x .venv/bin/python || uv venv --python 3.11 .venv
  mkdir -p .runtime/cad-wheels
  uv pip install --python .venv/bin/python -e "$PIPER_ROOT" \
    'cadquery-ocp==7.7.2' 'usd-core==26.8' --find-links "$PIPER_ROOT/.runtime/cad-wheels"
  .venv/bin/python scripts/convert_workcell_cad.py
fi
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PIPER_ROOT/src" "$PIPER_PYTHON" -m pytest -q
mkdir -p outputs
uv pip freeze --python "$PIPER_PYTHON" > outputs/runtime-freeze.txt
uv pip check --python "$PIPER_PYTHON"
echo "Installed. Launch: bash scripts/run.sh --headless --enable_cameras --steps 120 --demo"
