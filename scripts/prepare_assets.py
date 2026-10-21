#!/usr/bin/env python3
"""Copy and normalize assets without modifying the upstream robot description."""
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
from piper_sim.config import ROOT, load


def prepare(cfg):
    source = Path(cfg["robot"]["source"])
    original = source / "urdf/piper_description.urdf"
    output = ROOT / cfg["robot"]["urdf"]
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source / "meshes", output.parent / "meshes", dirs_exist_ok=True)
    tree = ET.parse(original)
    root = tree.getroot()
    # Remove the empty dummy root; importer fixes base_link to the world.
    root.remove(root.find("link[@name='dummy_link']"))
    root.remove(root.find("joint[@name='base_to_dummy']"))
    for mesh in root.iter("mesh"):
        mesh.set("filename", str((output.parent / "meshes" /
                                  Path(mesh.get("filename")).name).resolve()))
    for name in ("joint7", "joint8"):
        joint = root.find(f"joint[@name='{name}']")
        joint.set("type", "prismatic")
        # Finger origins differ by pi around local Z: +local Z moves them apart.
        ET.SubElement(joint, "axis", xyz="0 0 1")
        ET.SubElement(joint, "limit", lower="0", upper="0.035", effort="10", velocity="0.1")
        ET.SubElement(joint, "dynamics", damping="1", friction="0")
    # One physical gripper motor: finger 8 follows finger 7. Their local axes
    # already point in opposite world directions, so the multiplier is +1.
    ET.SubElement(root.find("joint[@name='joint8']"), "mimic",
                  joint="joint7", multiplier="1", offset="0")
    ET.indent(tree)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    provenance = {
        "source": str(original), "source_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "changes": ["remove empty dummy_link", "absolute mesh paths",
                    "restore prismatic fingers; joint8 mimics joint7; width=2*joint7"],
        "calibration": "URDF effort and dynamics not identified on hardware; mount calibration pending",
    }
    (output.parent / "provenance.json").write_text(json.dumps(provenance, indent=2))
    # Carry upstream licensing with copied geometry.
    license_path = source.parent / "LICENSE"
    if license_path.exists():
        shutil.copy2(license_path, output.parent / "UPSTREAM_LICENSE")
    print(output)


if __name__ == "__main__":
    prepare(load())
