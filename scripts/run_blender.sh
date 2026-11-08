#!/usr/bin/env bash
set -euo pipefail
blender_bin="${BLENDER_BIN:-}"
if [[ -z "$blender_bin" ]]; then
  blender_bin="$(command -v blender || true)"
fi
if [[ -z "$blender_bin" ]]; then
  blender_bin="$HOME/.local/opt/blender/blender-4.5.15-linux-x64/blender"
fi
if [[ ! -x "$blender_bin" ]]; then
  echo 'Blender not found. Run scripts/install_blender.sh or set BLENDER_BIN.' >&2
  exit 1
fi
exec "$blender_bin" "$@"
