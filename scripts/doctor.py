#!/usr/bin/env python3
"""Read-only environment report; does not launch Kit or connect to CAN."""
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys

packages = {}
for name in ("isaacsim", "isaaclab", "torch", "numpy", "h5py", "piper-sim"):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                      "--format=csv,noheader"], capture_output=True, text=True)
print(json.dumps({"python": sys.executable, "version": platform.python_version(),
                  "glibc": platform.libc_ver(), "packages": packages,
                  "gpu": gpu.stdout.strip(), "gpu_error": gpu.stderr.strip(),
                  "urdf_ready": (Path(__file__).resolve().parents[1] /
                                 "assets/generated/piper/piper.urdf").exists()},
                 indent=2, ensure_ascii=False))
