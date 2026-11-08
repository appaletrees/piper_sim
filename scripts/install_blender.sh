#!/usr/bin/env bash
# Pinned portable user installation. No system Python or sudo changes.
set -euo pipefail
blender_version=4.5.15
blender_archive="blender-${blender_version}-linux-x64.tar.xz"
blender_install="$HOME/.local/opt/blender"
blender_checksum=0d2b8bb2efe97e6c8eb57de6b7322f8800e1984e3125b60decca5f6771e4773e
mkdir -p "$blender_install"
curl -fL --retry 2 "https://download.blender.org/release/Blender4.5/$blender_archive" -o "$blender_install/$blender_archive"
printf '%s  %s\n' "$blender_checksum" "$blender_install/$blender_archive" | sha256sum --check --status
tar -xJf "$blender_install/$blender_archive" -C "$blender_install"
"$blender_install/blender-${blender_version}-linux-x64/blender" --version
