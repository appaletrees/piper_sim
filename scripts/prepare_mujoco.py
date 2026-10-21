#!/usr/bin/env python3
"""Build the MuJoCo model from the authoritative shared assets."""
import argparse
from pathlib import Path
from piper_sim.config import load
from piper_sim.mujoco_asset import build_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    path, manifest = build_model(load(args.config), args.output)
    print(f"{path}\n{len(manifest['cad_meshes'])} CAD meshes; source {manifest['source_cad_sha256']}")
